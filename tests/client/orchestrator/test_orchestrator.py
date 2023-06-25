import os
from unittest import mock
from unittest.mock import patch, MagicMock

import mongomock
import pandas as pd
from torch import nn

import multi_modal_edge_ai.client.databases.adl_queries as db_queries
from multi_modal_edge_ai.client.common.adl_model_keeper import ADLModelKeeper
from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper
from multi_modal_edge_ai.client.orchestrator import create_heartbeat_and_send, activity_is_finished, \
    initiate_internal_pipeline, start_federated_client, run_federation_stage, workload_lock, \
    start_personalization_client, run_personalization_stage
from multi_modal_edge_ai.commons.string_label_encoder import StringLabelEncoder
from multi_modal_edge_ai.models.adl_inference.ml_models.svm_model import SVMModel
from multi_modal_edge_ai.models.anomaly_detection.ml_models import Autoencoder


def test_create_heartbeat_and_send():
    root_directory = os.path.abspath(os.path.dirname(__file__))
    model_data_directory = os.path.join(root_directory, 'model_data')

    distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside',
                         'Movement']
    adl_encoder = StringLabelEncoder(distinct_adl_list)
    adl_model = SVMModel()
    adl_model_path = os.path.join(model_data_directory, 'adl_model')
    adl_encoder_path = os.path.join(model_data_directory, 'adl_encoder')
    adl_model_keeper = ADLModelKeeper(adl_model, adl_model_path, adl_encoder, adl_encoder_path)
    adl_model_keeper.num_predictions = 10

    anomaly_detection_model = Autoencoder([96, 64, 32, 24, 16, 8], [8, 16, 24, 32, 64, 96], nn.ReLU(), nn.Sigmoid())
    anomaly_detection_model_path = os.path.join(model_data_directory, 'adl_model')
    anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_detection_model_path)
    anomaly_detection_model_keeper.num_predictions = 5

    dictionary = {
        'adl_model_keeper': adl_model_keeper,
        'anomaly_detection_model_keeper': anomaly_detection_model_keeper
    }

    create_heartbeat_and_send(dictionary)

    # Assert the expected behavior
    assert adl_model_keeper.num_predictions == 0
    assert anomaly_detection_model_keeper.num_predictions == 0


def test_activity_is_finished_len_1_same_activity():
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection']

    # Store the original method
    original_get_past_x_activities = db_queries.get_past_x_activities

    # Mock the past_activity_list
    past_activity_list = [(pd.Timestamp("2023-06-01 10:00:00"), pd.Timestamp("2023-06-01 10:05:00"), 'ActivityAny')]
    db_queries.get_past_x_activities = mock.MagicMock(return_value=past_activity_list)

    # Add the past activity to the collection
    past_activity = {'Start_Time': pd.Timestamp("2023-06-01 10:00:00"),
                     'End_Time': pd.Timestamp("2023-06-01 10:05:00"), 'Activity': 'ActivityAny'}
    mock_collection.insert_one(past_activity)

    assert activity_is_finished("ActivityAny", mock_collection) is False

    # Restore the original method after the test
    db_queries.get_past_x_activities = original_get_past_x_activities


def test_activity_is_finished_len_1_different_activity():
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection']

    # Store the original method
    original_get_past_x_activities = db_queries.get_past_x_activities

    # Mock the past_activity_list
    past_activity_list = [(pd.Timestamp("2023-06-01 10:00:00"), pd.Timestamp("2023-06-01 10:05:00"), 'Activity')]
    db_queries.get_past_x_activities = mock.MagicMock(return_value=past_activity_list)

    # Add the past activity to the collection
    past_activity = {'Start_Time': pd.Timestamp("2023-06-01 10:00:00"),
                     'End_Time': pd.Timestamp("2023-06-01 10:05:00"), 'Activity': 'Activity'}
    mock_collection.insert_one(past_activity)

    assert activity_is_finished("ActivityAny", mock_collection) is True

    # Restore the original method after the test
    db_queries.get_past_x_activities = original_get_past_x_activities


def test_activity_is_finished_len_0():
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection']

    # Store the original method
    original_get_past_x_activities = db_queries.get_past_x_activities

    # Mock the past_activity_list
    past_activity_list = []
    db_queries.get_past_x_activities = mock.MagicMock(return_value=past_activity_list)

    assert activity_is_finished("ActivityAny", mock_collection) is True

    # Restore the original method after the test
    db_queries.get_past_x_activities = original_get_past_x_activities


def test_exception_initiate_internal_pipeline(caplog):
    initiate_internal_pipeline({'adl_model_keeper': None, 'anomaly_detection_model_keeper': None})
    # Check if the error message was logged
    expected_error_message = "An error occurred in the internal pipeline: 'sensor_db'"
    assert any(
        record.levelname == 'ERROR' and record.getMessage() == expected_error_message for record in caplog.records)


def test_initiate_internal_pipeline_no_adl_predicted(caplog):
    with patch('multi_modal_edge_ai.client.orchestrator.adl_inference_stage',
               return_value=None) as mock_adl_inference_stage:
        initiate_internal_pipeline({'adl_model_keeper': None, 'adl_window_size': 60, 'sensor_db': 'sensor_db'})
        expected_error_message = "An error occurred in the internal pipeline: No ADL predicted"
        assert any(
            record.levelname == 'ERROR' and record.getMessage() == expected_error_message for record in caplog.records)


def test_initiate_pipeline_activity_not_finished():
    with patch('multi_modal_edge_ai.client.orchestrator.adl_inference_stage') as mock_adl_inference_stage, \
            patch('multi_modal_edge_ai.client.orchestrator.get_database_client') as mock_get_db_client, \
            patch('multi_modal_edge_ai.client.orchestrator.activity_is_finished') as mock_activity_finished, \
            patch('multi_modal_edge_ai.client.orchestrator.add_activity') as mock_add_activity:
        mock_adl_inference_stage.return_value = {
            'Start_Time': pd.Timestamp('2023-06-22 10:00:00'),
            'End_Time': pd.Timestamp('2023-06-22 11:00:00'),
            'Activity': 'Relax'
        }
        mock_db_client = mongomock.MongoClient()
        mock_get_db_client.return_value = mock_db_client
        mock_activity_finished.return_value = False

        root_directory = os.path.abspath(os.path.dirname(__file__))
        model_data_directory = os.path.join(root_directory, 'model_data')

        distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside',
                             'Movement']
        adl_encoder = StringLabelEncoder(distinct_adl_list)
        adl_model = SVMModel()
        adl_model_path = os.path.join(model_data_directory, 'adl_model')
        adl_encoder_path = os.path.join(model_data_directory, 'adl_encoder')
        adl_model_keeper = ADLModelKeeper(adl_model, adl_model_path, adl_encoder, adl_encoder_path)
        adl_model_keeper.num_predictions = 10

        anomaly_detection_model = Autoencoder([96, 64, 32, 24, 16, 8], [8, 16, 24, 32, 64, 96], nn.ReLU(), nn.Sigmoid())
        anomaly_detection_model_path = os.path.join(model_data_directory, 'adl_model')
        anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_detection_model_path)
        anomaly_detection_model_keeper.num_predictions = 5

        client_config = {
            'adl_model_keeper': adl_model_keeper,
            'sensor_db': 'sensor_db_name',
            'adl_window_size': 60,
            'client_db': 'client_db_name',
            'adl_collection': 'adl_collection_name',
            'anomaly_collection': 'anomaly_collection_name',
            'anomaly_detection_window_size': 5,
            'anomaly_detection_model_keeper': anomaly_detection_model_keeper,
            'andet_scaler': None,
            'onehot_encoder': None,
            'num_adl_features': 3
        }

        initiate_internal_pipeline(client_config)

        mock_adl_inference_stage.assert_called_once()
        mock_get_db_client.assert_called_once()
        mock_activity_finished.assert_called_once()
        mock_add_activity.assert_called_once()
        assert adl_model_keeper.num_predictions == 10
        assert anomaly_detection_model_keeper.num_predictions == 5


def test_initiate_pipeline_activity_finished_not_anomalous():
    with patch('multi_modal_edge_ai.client.orchestrator.adl_inference_stage') as mock_adl_inference_stage, \
            patch('multi_modal_edge_ai.client.orchestrator.get_database_client') as mock_get_db_client, \
            patch('multi_modal_edge_ai.client.orchestrator.activity_is_finished') as mock_activity_finished, \
            patch('multi_modal_edge_ai.client.orchestrator.check_window_for_anomaly') as mock_check_anomaly, \
            patch('multi_modal_edge_ai.client.orchestrator.add_activity') as mock_add_activity:
        mock_adl_inference_stage.return_value = {
            'Start_Time': pd.Timestamp('2023-06-22 10:00:00'),
            'End_Time': pd.Timestamp('2023-06-22 11:00:00'),
            'Activity': 'Relax'
        }
        mock_db_client = mongomock.MongoClient()
        mock_get_db_client.return_value = mock_db_client
        mock_activity_finished.return_value = True
        mock_check_anomaly.return_value = 1

        root_directory = os.path.abspath(os.path.dirname(__file__))
        model_data_directory = os.path.join(root_directory, 'model_data')

        distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside',
                             'Movement']
        adl_encoder = StringLabelEncoder(distinct_adl_list)
        adl_model = SVMModel()
        adl_model_path = os.path.join(model_data_directory, 'adl_model')
        adl_encoder_path = os.path.join(model_data_directory, 'adl_encoder')
        adl_model_keeper = ADLModelKeeper(adl_model, adl_model_path, adl_encoder, adl_encoder_path)
        adl_model_keeper.num_predictions = 10

        anomaly_detection_model = Autoencoder([96, 64, 32, 24, 16, 8], [8, 16, 24, 32, 64, 96], nn.ReLU(), nn.Sigmoid())
        anomaly_detection_model_path = os.path.join(model_data_directory, 'adl_model')
        anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_detection_model_path)
        anomaly_detection_model_keeper.num_predictions = 5

        client_config = {
            'adl_model_keeper': adl_model_keeper,
            'sensor_db': 'sensor_db_name',
            'adl_window_size': 60,
            'client_db': 'client_db_name',
            'adl_collection': 'adl_collection_name',
            'anomaly_collection': 'anomaly_collection_name',
            'anomaly_detection_window_size': 5,
            'anomaly_detection_model_keeper': anomaly_detection_model_keeper,
            'andet_scaler': None,
            'onehot_encoder': None,
            'num_adl_features': 3
        }

        initiate_internal_pipeline(client_config)

        mock_adl_inference_stage.assert_called_once()
        mock_get_db_client.assert_called_once()
        mock_activity_finished.assert_called_once()
        mock_check_anomaly.assert_called_once()
        mock_add_activity.assert_called_once()
        assert adl_model_keeper.num_predictions == 11
        assert anomaly_detection_model_keeper.num_predictions == 5


def test_initiate_pipeline_activity_finished_anomalous():
    with patch('multi_modal_edge_ai.client.orchestrator.adl_inference_stage') as mock_adl_inference_stage, \
            patch('multi_modal_edge_ai.client.orchestrator.get_database_client') as mock_get_db_client, \
            patch('multi_modal_edge_ai.client.orchestrator.activity_is_finished') as mock_activity_finished, \
            patch('multi_modal_edge_ai.client.orchestrator.check_window_for_anomaly') as mock_check_anomaly, \
            patch('multi_modal_edge_ai.client.orchestrator.add_activity') as mock_add_activity:
        mock_adl_inference_stage.return_value = {
            'Start_Time': pd.Timestamp('2023-06-22 10:00:00'),
            'End_Time': pd.Timestamp('2023-06-22 11:00:00'),
            'Activity': 'Relax'
        }
        mock_db_client = mongomock.MongoClient()
        mock_get_db_client.return_value = mock_db_client
        mock_activity_finished.return_value = True
        mock_check_anomaly.return_value = 0

        root_directory = os.path.abspath(os.path.dirname(__file__))
        model_data_directory = os.path.join(root_directory, 'model_data')

        distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside',
                             'Movement']
        adl_encoder = StringLabelEncoder(distinct_adl_list)
        adl_model = SVMModel()
        adl_model_path = os.path.join(model_data_directory, 'adl_model')
        adl_encoder_path = os.path.join(model_data_directory, 'adl_encoder')
        adl_model_keeper = ADLModelKeeper(adl_model, adl_model_path, adl_encoder, adl_encoder_path)
        adl_model_keeper.num_predictions = 10

        anomaly_detection_model = Autoencoder([96, 64, 32, 24, 16, 8], [8, 16, 24, 32, 64, 96], nn.ReLU(), nn.Sigmoid())
        anomaly_detection_model_path = os.path.join(model_data_directory, 'adl_model')
        anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_detection_model_path)
        anomaly_detection_model_keeper.num_predictions = 5

        client_config = {
            'adl_model_keeper': adl_model_keeper,
            'sensor_db': 'sensor_db_name',
            'adl_window_size': 60,
            'client_db': 'client_db_name',
            'adl_collection': 'adl_collection_name',
            'anomaly_collection': 'anomaly_collection_name',
            'anomaly_detection_window_size': 5,
            'anomaly_detection_model_keeper': anomaly_detection_model_keeper,
            'andet_scaler': None,
            'onehot_encoder': None,
            'num_adl_features': 3
        }

        initiate_internal_pipeline(client_config)

        mock_adl_inference_stage.assert_called_once()
        mock_get_db_client.assert_called_once()
        mock_activity_finished.assert_called_once()
        mock_check_anomaly.assert_called_once()
        mock_add_activity.assert_called_once()
        assert adl_model_keeper.num_predictions == 11
        assert anomaly_detection_model_keeper.num_predictions == 6


def test_start_federated_client():
    with patch('multi_modal_edge_ai.client.orchestrator.get_database_client') as mock_get_db_client, \
            patch('multi_modal_edge_ai.client.orchestrator.TrainEval') as mock_train_eval, \
            patch('multi_modal_edge_ai.client.orchestrator.FederatedClient') as mock_fc, \
            patch('multi_modal_edge_ai.client.orchestrator.workload_lock') as mock_workload_lock:
        distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside',
                             'Movement']

        mock_db_client = mongomock.MongoClient()
        mock_get_db_client.return_value = mock_db_client

        mock_train_eval_instance = MagicMock()
        mock_train_eval.return_value = mock_train_eval_instance

        mock_fc_instance = MagicMock()
        mock_fc.return_value = mock_fc_instance

        client_config = {
            'adl_model_keeper': None,
            'sensor_db': 'sensor_db_name',
            'adl_window_size': 60,
            'adl_list': distinct_adl_list,
            'client_db': 'client_db_name',
            'adl_collection': 'adl_collection_name',
            'anomaly_collection': 'anomaly_collection_name',
            'anomaly_detection_window_size': 5,
            'anomaly_detection_model_keeper': None,
            'andet_scaler': None,
            'onehot_encoder': None,
            'num_adl_features': 3
        }

        start_federated_client(client_config)

        mock_workload_lock.release.assert_called_once()
        mock_get_db_client.assert_called_once()
        mock_train_eval.assert_called_once()
        mock_fc.assert_called_once()
        mock_fc_instance.start_numpy_client.assert_called_once_with("127.0.0.1:8080")


def test_run_federation_stage():
    with patch('multi_modal_edge_ai.client.orchestrator.start_federated_client') as mock_start_federated_client, \
            patch('multi_modal_edge_ai.client.orchestrator.threading.Thread') as mock_thread, \
            patch('multi_modal_edge_ai.client.orchestrator.workload_lock') as mock_workload_lock:
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        mock_workload_lock.acquire.return_value = True

        distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside',
                             'Movement']
        client_config = {
            'adl_model_keeper': None,
            'sensor_db': 'sensor_db_name',
            'adl_window_size': 60,
            'adl_list': distinct_adl_list,
            'client_db': 'client_db_name',
            'adl_collection': 'adl_collection_name',
            'anomaly_collection': 'anomaly_collection_name',
            'anomaly_detection_window_size': 5,
            'anomaly_detection_model_keeper': None,
            'andet_scaler': None,
            'onehot_encoder': None,
            'num_adl_features': 3
        }

        run_federation_stage(client_config)

        mock_thread.assert_called_once_with(target=mock_start_federated_client, args=[client_config])
        mock_thread_instance.start.assert_called_once()


def test_start_personalization_client():
    with patch('multi_modal_edge_ai.client.orchestrator.get_database_client') as mock_get_db_client, \
            patch('multi_modal_edge_ai.client.orchestrator.TrainEval') as mock_train_eval, \
            patch('multi_modal_edge_ai.client.orchestrator.FederatedClient') as mock_fc, \
            patch('multi_modal_edge_ai.client.orchestrator.workload_lock') as mock_workload_lock:
        distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside',
                             'Movement']

        mock_db_client = mongomock.MongoClient()
        mock_get_db_client.return_value = mock_db_client

        mock_train_eval_instance = MagicMock()
        mock_train_eval.return_value = mock_train_eval_instance

        mock_fc_instance = MagicMock()
        mock_fc.return_value = mock_fc_instance

        client_config = {
            'adl_model_keeper': None,
            'sensor_db': 'sensor_db_name',
            'adl_window_size': 60,
            'adl_list': distinct_adl_list,
            'client_db': 'client_db_name',
            'adl_collection': 'adl_collection_name',
            'anomaly_collection': 'anomaly_collection_name',
            'anomaly_detection_window_size': 5,
            'anomaly_detection_model_keeper': None,
            'andet_scaler': None,
            'onehot_encoder': None,
            'num_adl_features': 3
        }

        start_personalization_client(client_config)

        mock_workload_lock.release.assert_called_once()
        mock_get_db_client.assert_called_once()
        mock_train_eval.assert_called_once()
        mock_fc.assert_called_once()
        mock_fc_instance.start_numpy_client.assert_called_once_with("127.0.0.1:8080")

def test_run_personalization_stage():
    with patch('multi_modal_edge_ai.client.orchestrator.start_personalization_client') \
            as mock_start_personalized_client, patch('multi_modal_edge_ai.client.orchestrator.threading.Thread') \
            as mock_thread, patch('multi_modal_edge_ai.client.orchestrator.workload_lock') as mock_workload_lock:
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        mock_workload_lock.acquire.return_value = True

        distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside',
                             'Movement']
        client_config = {
            'adl_model_keeper': None,
            'sensor_db': 'sensor_db_name',
            'adl_window_size': 60,
            'adl_list': distinct_adl_list,
            'client_db': 'client_db_name',
            'adl_collection': 'adl_collection_name',
            'anomaly_collection': 'anomaly_collection_name',
            'anomaly_detection_window_size': 5,
            'anomaly_detection_model_keeper': None,
            'andet_scaler': None,
            'onehot_encoder': None,
            'num_adl_features': 3
        }

        run_personalization_stage(client_config)

        mock_thread.assert_called_once_with(target=mock_start_personalized_client, args=[client_config])
        mock_thread_instance.start.assert_called_once()
