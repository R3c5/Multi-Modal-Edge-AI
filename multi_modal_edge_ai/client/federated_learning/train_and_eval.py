from typing import Dict, Optional

import pandas as pd
import torch
from flwr.common import Scalar
from pymongo.collection import Collection
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, log_loss
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from multi_modal_edge_ai.client.databases.adl_queries import get_all_activities
from multi_modal_edge_ai.commons.model import Model
from multi_modal_edge_ai.models.anomaly_detection.data_access.adl_dataset import ADLDataset
from multi_modal_edge_ai.models.anomaly_detection.preprocessing.adl_dataframe_preprocessing import (
    dataframe_categorical_to_numeric
)
from multi_modal_edge_ai.models.anomaly_detection.preprocessing.window_splitter import split_into_windows
from multi_modal_edge_ai.models.anomaly_detection.train_and_eval.model_validator import validation_df_concatenation
from multi_modal_edge_ai.models.anomaly_detection.train_and_eval.synthetic_anomaly_generator import (
    clean_windows,
    synthetic_anomaly_generator
)


class TrainEval:

    def __init__(self, database_collection: Collection, distinct_adl_list: list[str], scaler):
        """
        Constructor for the train eval object.
        :param database_collection: The database collection respective of this client
        :param distinct_adl_list: A list with the distinct ADLs present in the data
        :param scaler: The scaler used to scale the data appropriately
        """
        self.database_collection = database_collection
        self.distinct_adl_list = distinct_adl_list
        self.scaler = scaler
        self.clean_test_windows: Optional[pd.DataFrame] = None
        self.anomalous_windows: Optional[pd.DataFrame] = None
        self.clean_train_windows: Optional[pd.DataFrame] = None

    def train(self, model: Model, config) -> tuple[int, Dict[str, Scalar]]:
        """
        This function will perform the training loop for given model and config.
        It will first read all the ADLs from the database, perform the windowing process, split the windows into clean
        and anomalous windows, store the anomalous for the later evaluate stage, split the clean windows into train and
        test datasets, perform categorical to numerical transformation, and scale the data according to the
        class specified scaler. Finally, it will train the model and return the size of the train dataset as well as the
        average training reconstruction loss incurred.
        :param model: The machine learning model to train
        :param config: The configuration parameters of the training procedure
        :return: The train dataset size and average training reconstruction loss
        """
        self.define_train_test(config)
        dataset_size = self.clean_train_windows.shape[0] if self.clean_train_windows is not None else 0

        numeric_training_df, n_features_adl = \
            dataframe_categorical_to_numeric(self.clean_train_windows, int(config["window_size"]),
                                             self.distinct_adl_list, config["one-hot"])

        normalized_training_df = transform_with_scaler(self.scaler, numeric_training_df, n_features_adl)

        normalized_training_dataloader = DataLoader(ADLDataset(normalized_training_df), config["batch_size"],
                                                    shuffle=True)

        training_loss = [loss.cpu().item() for loss in model.train(normalized_training_dataloader, **config)]

        return (dataset_size, {}) if training_loss is None else \
            (dataset_size, {"avg_reconstruction_loss": sum(training_loss) / len(training_loss)})

    def evaluate(self, model: Model, config: dict[str, Scalar]) -> tuple[float, int, Dict[str, Scalar]]:
        """
        This method takes a model and a configuration dictionary as input. It generates synthetic anomalies, prepares
        the testing dataset, transforms it using a scaler, and uses the model to make predictions. Afterwards, it
        computes the accuracy, recall, precision, and F1-score metrics based on the model's predictions and the actual
        labels.
        It also calculates the log loss. If a training test split wasn't yet performed on this object, it will force
        the train test split.
        :param model: The machine learning model to train
        :param config: The configuration parameters of the evaluation procedure
        :return: The log loss of the evaluation, the size of the evaluation dataset, and a dict with accuracy, recall,
        precision, and f1-score
        """
        if self.clean_test_windows is None or self.anomalous_windows is None:
            self.define_train_test(config)

        generated_anomalies_df = synthetic_anomaly_generator(self.anomalous_windows,
                                                             float(config["anomaly_generation_ratio"]))
        anomalous_df = pd.concat([self.anomalous_windows, generated_anomalies_df])

        testing_df = self.clean_test_windows.copy() if self.clean_test_windows is not None else pd.DataFrame([])
        testing_df, testing_labels = validation_df_concatenation(testing_df, anomalous_df)
        dataset_size = testing_df.shape[0]

        numeric_testing_df, n_features_adl = dataframe_categorical_to_numeric(testing_df, int(config["window_size"]),
                                                                              self.distinct_adl_list,
                                                                              bool(config["one-hot"]))

        normalized_testing_df = transform_with_scaler(self.scaler, numeric_testing_df, n_features_adl)

        if hasattr(model, 'set_reconstruction_error_threshold'):
            model.set_reconstruction_error_threshold(config["reconstruction_error_quantile"])

        numpy_windows = normalized_testing_df.values
        predicted_labels = [model.predict(torch.Tensor(window)) for window in numpy_windows]

        accuracy = accuracy_score(testing_labels, predicted_labels)
        recall = recall_score(testing_labels, predicted_labels)
        precision = precision_score(testing_labels, predicted_labels)
        f1 = f1_score(testing_labels, predicted_labels)
        loss = log_loss(testing_labels, predicted_labels)

        return loss, dataset_size, {"accuracy": accuracy, "recall": recall, "precision": precision, "f1_score": f1}

    def define_train_test(self, config: dict[str, Scalar]) -> None:
        """
        This function will read the adls from the database, and define train and test data with regards to the given
        parameters.
        :param config: The training/evaluation config
        :return: None
        """
        data = pd.DataFrame(get_all_activities(self.database_collection))
        data.columns = ["Start_Time", "End_Time", "Activity"]

        window_df = split_into_windows(data, float(config["window_size"]), float(config["window_slide"]),
                                       bool(config["event_based"]))

        clean_df, self.anomalous_windows = clean_windows(data, window_df, float(config["anomaly_whisker"]))
        clean_df = shuffle(clean_df, random_state=42)

        split_index = int(len(clean_df) * config["clean_test_data_ratio"])
        self.clean_test_windows = clean_df[:split_index]
        self.clean_train_windows = clean_df[split_index:]


def transform_with_scaler(scaler, windowed_adl_df, n_features):
    """
    This function will perform scaling according to the passed scaler. However, it is important to note that it will
    perform the scaling according to each ADL, and not to each row. Therefore, it requires the number of features per
    ADL so that it can be reshaped to one ADL per row, perform the scaling, and reshape back again to original shape
    :param scaler: The scaler provided
    :param windowed_adl_df: The dataframe with the windows
    :param n_features: The number of features per ADL
    :return: the rescaled dataframe
    """
    original_shape = windowed_adl_df.shape

    reshaped_df = pd.DataFrame(windowed_adl_df.values.reshape((-1, n_features)))
    rescaled_df = scaler.transform(reshaped_df)

    rescaled_array = rescaled_df.values.reshape(original_shape)

    return pd.DataFrame(rescaled_array, columns=windowed_adl_df.columns,
                        index=windowed_adl_df.index)
