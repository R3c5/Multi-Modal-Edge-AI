from unittest.mock import patch, Mock

import pandas as pd
import torch.nn
from pymongo.collection import Collection

from multi_modal_edge_ai.client.federated_learning.train_and_eval import TrainEval, transform_with_scaler
from multi_modal_edge_ai.models.anomaly_detection.data_access.parser import parse_file_with_idle
from multi_modal_edge_ai.models.anomaly_detection.ml_models import Autoencoder
from multi_modal_edge_ai.models.anomaly_detection.preprocessing.adl_dataframe_preprocessing import \
    dataframe_categorical_to_numeric
from multi_modal_edge_ai.models.anomaly_detection.preprocessing.window_splitter import split_into_windows


def training_fixture(get_all_activities_mock):
    adl_dataset = parse_file_with_idle(
        "tests/models/anomaly_detection/dummy_datasets/dummy_aruba.csv")
    distinct_adl_list = pd.unique(adl_dataset.iloc[:, 2::3].values.ravel('K'))

    database_collection = Mock(spec=Collection)
    scaler = Mock()
    get_all_activities_mock.side_effect = lambda collection: adl_dataset

    train_eval = TrainEval(database_collection, distinct_adl_list, scaler)

    config = {
        "window_size": 10,
        "window_slide": 8,
        "event_based": True,
        "anomaly_whisker": 1.5,
        "clean_test_data_ratio": 0.2,
        "one-hot": True,
        "batch_size": 32,
        "learning_rate": 0.01,
        "n_epochs": 2,
        "verbose": False
    }

    model = Autoencoder([120, 50], [50, 120], torch.nn.ReLU(), torch.nn.Sigmoid())

    return model, train_eval, config


@patch('multi_modal_edge_ai.client.federated_learning.train_and_eval.transform_with_scaler',
       side_effect=lambda sc, df, n_feat: df)
@patch('multi_modal_edge_ai.client.federated_learning.train_and_eval.get_all_activities')
def test_simple_training(get_all_activities_mock, transform_with_scaler_mock):
    model, train_eval, config = training_fixture(get_all_activities_mock)
    df_size, metrics = train_eval.train(model, config)

    assert df_size == 77
    assert metrics["avg_reconstruction_loss"] is not None


@patch('multi_modal_edge_ai.client.federated_learning.train_and_eval.transform_with_scaler',
       side_effect=lambda sc, df, n_feat: df)
@patch('multi_modal_edge_ai.client.federated_learning.train_and_eval.get_all_activities')
def test_evaluate_simple(get_all_activities_mock, transform_with_scaler_mock):
    model, train_eval, config = training_fixture(get_all_activities_mock)

    config.update({
        "anomaly_generation_ratio": 0.1,
        "reconstruction_error_quantile": 0.99
    })

    loss, df_size, metrics = train_eval.evaluate(model, config)

    assert df_size == 51  # TODO Change when synthetic anomaly generation is fixed
    assert isinstance(loss, float)

    expected_keys = {"accuracy", "recall", "precision", "f1_score"}

    assert set(metrics.keys()) == expected_keys
    assert all(isinstance(value, float) for value in metrics.values())


@patch('multi_modal_edge_ai.client.federated_learning.train_and_eval.get_all_activities')
def test_define_datasets(get_all_activities_mock):
    _, train_eval, config = training_fixture(get_all_activities_mock)

    train_eval.define_train_test(config)

    assert train_eval.clean_train_windows.shape[0] == 77
    assert train_eval.clean_test_windows.shape[0] == 19
    assert train_eval.anomalous_windows.shape[0] == 16


def test_transform_scaler():
    scaler = Mock()
    scaler.transform.side_effect = lambda x: x * 2

    adl_dataset = parse_file_with_idle(
        "tests/models/anomaly_detection/dummy_datasets/dummy_aruba.csv")
    distinct_adl_list = pd.unique(adl_dataset.iloc[:, 2::3].values.ravel('K'))

    window_df = split_into_windows(adl_dataset, 10, 8, True)

    numeric_training_df, n_features_adl = dataframe_categorical_to_numeric(window_df, 10, distinct_adl_list, False)

    returned_df = transform_with_scaler(scaler, numeric_training_df, n_features_adl)
    expected_df = numeric_training_df.apply(lambda x: x * 2)

    pd.testing.assert_frame_equal(expected_df, returned_df)
