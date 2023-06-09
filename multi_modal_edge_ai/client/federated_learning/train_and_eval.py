from typing import Dict

import pandas as pd
import torch
from flwr.common import Scalar
from pymongo.collection import Collection
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, log_loss
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from multi_modal_edge_ai.client.adl_database.adl_queries import get_all_activities
from multi_modal_edge_ai.commons.model import Model
from multi_modal_edge_ai.models.anomaly_detection.data_access.adl_dataset import ADLDataset
from multi_modal_edge_ai.models.anomaly_detection.preprocessing.adl_dataframe_preprocessing import \
    dataframe_categorical_to_numeric
from multi_modal_edge_ai.models.anomaly_detection.preprocessing.window_splitter import split_into_windows
from multi_modal_edge_ai.models.anomaly_detection.train_and_eval.model_validator import validation_df_concatenation
from multi_modal_edge_ai.models.anomaly_detection.train_and_eval.synthetic_anomaly_generator import clean_windows, \
    synthetic_anomaly_generator


class TrainEval:

    def __init__(self, database_collection: Collection, distinct_adl_list: list[str], scaler):
        self.database_collection = database_collection
        self.distinct_adl_list = distinct_adl_list
        self.scaler = scaler
        self.clean_test_windows = None
        self.anomalous_windows = None

    def train(self, model: Model, config) -> tuple[int, Dict[str, Scalar]]:
        data = pd.DataFrame(get_all_activities(self.database_collection))

        window_df = split_into_windows(data, config["window_size"], config["window_slide"], config["event_based"])

        clean_df, self.anomalous_windows = clean_windows(data, window_df, config["anomaly_whisker"])
        clean_df = shuffle(clean_df)

        split_index = int(len(clean_df) * config["clean_test_data_ratio"])
        self.clean_test_windows = clean_df[:split_index]
        training_df = clean_df[split_index:]
        df_size = training_df.shape[0]

        numeric_training_df, n_features_adl = dataframe_categorical_to_numeric(training_df, int(config["window_size"]),
                                                                               self.distinct_adl_list,
                                                                               config["one-hot"])

        normalized_training_df = transform_with_scaler(self.scaler, numeric_training_df, n_features_adl)

        normalized_training_dataloader = DataLoader(ADLDataset(normalized_training_df), config["batch_size"],
                                                    shuffle=True)

        training_loss = model.train(normalized_training_dataloader, **config)

        return (df_size, {}) if training_loss is None else \
            (df_size, {"avg_reconstruction_loss": sum(training_loss) / len(training_loss)})

    def evaluate(self, model: Model, config: dict[str, Scalar]) -> tuple[float, int, Dict[str, Scalar]]:
        generated_anomalies_df = synthetic_anomaly_generator(self.anomalous_windows, config["anomaly_generation_ratio"])
        anomalous_df = pd.concat([self.anomalous_windows, generated_anomalies_df])

        testing_df = self.clean_test_windows.copy()
        testing_df, testing_labels = validation_df_concatenation(testing_df, anomalous_df)
        df_size = testing_df.shape[0]

        numeric_testing_df, n_features_adl = dataframe_categorical_to_numeric(testing_df, int(config["window_size"]),
                                                                              self.distinct_adl_list, config["one-hot"])

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

        return loss, df_size, {"accuracy": accuracy, "recall": recall, "precision": precision, "f1_score": f1}


def transform_with_scaler(scaler, windowed_adl_df, n_features):
    original_shape = windowed_adl_df.shape

    reshaped_df = pd.DataFrame(windowed_adl_df.values.reshape((-1, n_features)))
    rescaled_df = scaler.transform(reshaped_df)

    return pd.DataFrame(rescaled_df.reshape(original_shape), columns=windowed_adl_df.columns,
                        index=windowed_adl_df.index)
