from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from multi_modal_edge_ai.commons.model import Model
from multi_modal_edge_ai.models.anomaly_detection.data_access.adl_dataset import ADLDataset
from multi_modal_edge_ai.models.anomaly_detection.preprocessing.adl_dataframe_preprocessing import \
    dataframe_categorical_to_numeric, dataframe_standard_scaling
from multi_modal_edge_ai.models.anomaly_detection.preprocessing.window_splitter import split_into_windows
from multi_modal_edge_ai.models.anomaly_detection.train_and_eval.hyperparameter_config import HyperparameterConfig
from multi_modal_edge_ai.models.anomaly_detection.train_and_eval.synthetic_anomaly_generator import clean_windows, \
    synthetic_anomaly_generator


def model_train_eval(model: Model, data: pd.DataFrame, hparams: HyperparameterConfig) -> tuple[float, Any]:
    """
    This function is responsible for training the model on the given dataset and then evaluating it. It splits the
    provided dataset into windows, cleans the windows, and generates synthetic anomalies. It also separates clean and
    anomalous instances, shuffles and splits the clean data for training and testing. It normalizes the training and
    testing data, transforms them to numerical form and prepares them for the model. The function also sets the
    reconstruction error threshold for the model if applicable. Finally, it predicts the labels for the testing dataset
    and calculates the accuracy and confusion matrix. Important to note that this function currently does not support
    time-based windows.
    :param model: The machine learning model that implements the Model abstract class
    :param data: The dataframe containing the ADLs sequences
    :param hparams: The HyperparameterConfig with all the hyperparameters for the model, training and validation
    procedure
    :return: A tuple with the first element being the accuracy and the second element being a confusion matrix of the
    prediction
    """
    window_df = split_into_windows(data, hparams.window_size, hparams.window_slide, hparams.event_based)

    distinct_adl_list = pd.unique(window_df.iloc[:, 2::3].values.ravel('K'))

    clean_df, anomalous_df = clean_windows(data, window_df, hparams.anomaly_whisker)

    clean_df = shuffle(clean_df)
    split_index = int(len(clean_df) * hparams.clean_test_data_ratio)
    testing_df = clean_df[:split_index]
    training_df = clean_df[split_index:]

    generated_anomalies_df = synthetic_anomaly_generator(anomalous_df, hparams.anomaly_generation_ratio)
    anomalous_df = pd.concat([anomalous_df, generated_anomalies_df])

    testing_df = testing_df.copy()
    testing_df, testing_labels = validation_df_concatenation(testing_df, anomalous_df)

    # currently this numerical transformation doesn't support time-based windows
    numeric_training_df, n_features_adl = dataframe_categorical_to_numeric(training_df, int(hparams.window_size),
                                                                           distinct_adl_list, hparams.one_hot)

    numeric_testing_df = \
        dataframe_categorical_to_numeric(testing_df, int(hparams.window_size), distinct_adl_list, hparams.one_hot)[0]

    # currently this numerical transformation doesn't support time-based windows
    normalized_training_df = dataframe_standard_scaling(numeric_training_df, n_features_adl)
    normalized_testing_df = dataframe_standard_scaling(numeric_testing_df, n_features_adl)

    normalized_training_dataloader = DataLoader(ADLDataset(normalized_training_df), hparams.batch_size, shuffle=True)

    model.train(normalized_training_dataloader, **vars(hparams))

    if hasattr(model, 'set_reconstruction_error_threshold'):
        model.set_reconstruction_error_threshold(hparams.reconstruction_error_quantile)

    numpy_windows = normalized_testing_df.values
    predicted_labels = [model.predict(torch.Tensor(window)) for window in numpy_windows]
    n_correctly_predicted = np.sum(testing_labels == np.array(predicted_labels))

    accuracy = n_correctly_predicted / len(testing_labels)
    cm = confusion_matrix(testing_labels, predicted_labels)

    return accuracy, cm


def validation_df_concatenation(clean_df: pd.DataFrame, anomalous_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    This function will perform the concatenation of both a clean testing dataframe and an anomalous testing dataframe
    to form a larger, comprehensive (with both classes) testing dataframe. It will also return a numpy array with
    the labels (1 for inlier, 0 for outlier) of each of the indices.
    :param clean_df: The dataframe with the clean data to be used for the testing dataframe
    :param anomalous_df: The dataframe with the anomalous data to be used for the testing dataframe
    :return: The testing dataframe and its respective labels.
    """
    clean_df.loc[:, "label"] = 1
    anomalous_df.loc[:, "label"] = 0
    testing_df = shuffle(pd.concat([anomalous_df, clean_df]))

    testing_labels = testing_df["label"].values
    testing_df = testing_df.drop("label", axis=1)
    testing_df = testing_df.drop("Reason", axis=1)
    testing_df = testing_df.drop("Duration", axis=1)

    return testing_df, testing_labels
