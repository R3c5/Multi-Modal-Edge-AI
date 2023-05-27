import pandas as pd

import multi_modal_edge_ai.anomaly_detection.ml_models.one_class_svm
from multi_modal_edge_ai.anomaly_detection.data_access.adl_dataset import ADLDataset
from multi_modal_edge_ai.anomaly_detection.preprocessing.adl_dataframe_preprocessing import \
    dataframe_categorical_to_numeric, dataframe_standard_scaling
from multi_modal_edge_ai.anomaly_detection.preprocessing.window_splitter import split_into_windows
from multi_modal_edge_ai.anomaly_detection.train_and_eval.hyperparameter_config import HyperparameterConfig
from multi_modal_edge_ai.commons.model import Model
from torch.utils.data import DataLoader


# window the dataframe
# clean and fabricate anomalies -> clean, anomalous
# Adjust training and testing set dimensions
# Convert to numerical data
# Train the model on the training data
# Test the model on the test data
# Report performances


def synthetic_anomaly_generation(window_df, param):
    return window_df, window_df


def model_train_eval(model: Model, data: pd.DataFrame, hparams: HyperparameterConfig):
    window_df = split_into_windows(data, hparams.window_size, hparams.window_slide, hparams.event_based)

    clean_df, anomalous_df = synthetic_anomaly_generation(window_df, ...)  # TODO clean & generate anomalies

    training_df, testing_df = clean_df, anomalous_df  # TODO split into train and test datasets. Requires synth anom.

    # currently this numerical transformation doesn't support time-based windows
    numeric_training_df, n_features_adl = dataframe_categorical_to_numeric(training_df, int(hparams.window_size),
                                                                           hparams.one_hot)
    numeric_testing_df = dataframe_categorical_to_numeric(testing_df, int(hparams.window_size), hparams.one_hot)[0]

    # currently this numerical transformation doesn't support time-based windows
    normalized_training_df = dataframe_standard_scaling(numeric_training_df, n_features_adl)
    normalized_testing_df = dataframe_standard_scaling(numeric_testing_df, n_features_adl)

    normalized_training_dataloader = DataLoader(ADLDataset(normalized_testing_df))
    normalized_testing_dataloader = DataLoader(ADLDataset(normalized_testing_df))

    model.train(normalized_training_dataloader, **vars(hparams))

