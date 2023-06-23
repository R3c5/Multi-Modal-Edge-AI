import logging
from unittest import mock
from unittest.mock import patch

import pytest
import requests
from torch import nn

from multi_modal_edge_ai.client.common.adl_model_keeper import ADLModelKeeper
from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper
from multi_modal_edge_ai.client.controllers.client_controller import send_set_up_connection_request, send_heartbeat, \
    save_models_zip_file, save_model_file
from multi_modal_edge_ai.commons.string_label_encoder import StringLabelEncoder
from multi_modal_edge_ai.models.adl_inference.ml_models.svm_model import SVMModel
from multi_modal_edge_ai.models.anomaly_detection.ml_models import Autoencoder

@pytest.fixture(autouse=True)
def set_logging_level(caplog):
    caplog.set_level(logging.INFO)

@pytest.fixture(scope="module")
def client_config():
    distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside',
                         'Movement']
    adl_encoder = StringLabelEncoder(distinct_adl_list)
    adl_model = SVMModel()
    adl_model_keeper_path = 'tests/client/controllers/models_test_files/test_model_dest'
    adl_encoder_path = 'tests/client/model_data/adl_encoder'
    adl_model_keeper = ADLModelKeeper(adl_model, adl_model_keeper_path, adl_encoder, adl_encoder_path)

    anomaly_detection_model = Autoencoder([96, 64, 32, 24, 16, 8], [8, 16, 24, 32, 64, 96], nn.ReLU(), nn.Sigmoid())
    anomaly_det_model_keeper_path = 'tests/client/controllers/models_test_files/test_model_dest'
    anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_det_model_keeper_path)

    return {
        'adl_model_keeper': adl_model_keeper,
        'anomaly_detection_model_keeper': anomaly_detection_model_keeper
    }


def test_send_set_up_connection_request_success(client_config, caplog):
    caplog.set_level(logging.INFO)
    with mock.patch.object(requests, 'get') as mock_get:
        mock_response = mock.Mock()
        mock_response.status_code = 200

        with mock.patch('multi_modal_edge_ai.client.controllers.client_controller.save_models_zip_file') \
                as mock_save_zip:
            mock_get.return_value = mock_response
            # Call the method under test
            send_set_up_connection_request(client_config['adl_model_keeper'],
                                           client_config['anomaly_detection_model_keeper'])
            # Assert that the appropriate functions were called
            mock_save_zip.assert_called_with(mock_get.return_value, client_config['adl_model_keeper'],
                                             client_config['anomaly_detection_model_keeper'])

            assert 'Connection set up successfully' in caplog.text


def test_send_set_up_connection_request_fail(client_config, caplog):
    with mock.patch.object(requests, 'get'):
        mock_response = mock.Mock()
        mock_response.status_code = 500
        mock_response.text = "Server error"

        send_set_up_connection_request(client_config['adl_model_keeper'],
                                       client_config['anomaly_detection_model_keeper'])

        # Check the log
        assert "An error occurred during set up with server: " in caplog.text


def test_send_heartbeat_success(client_config, caplog):
    with mock.patch.object(requests, 'post') as mock_post:
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.headers = {'start_federation_client_flag': 'False'}

        with mock.patch('multi_modal_edge_ai.client.controllers.client_controller.save_models_zip_file') \
                as mock_save_zip:
            mock_post.return_value = mock_response
            # Call the method under test
            send_heartbeat(client_config, 0, 0)
            # Assert that the appropriate functions were called
            mock_save_zip.assert_called_with(mock_post.return_value, client_config['adl_model_keeper'],
                                             client_config['anomaly_detection_model_keeper'])

            assert 'Heartbeat successful' in caplog.text


def test_send_heartbeat_no_client(client_config, caplog):
    with mock.patch.object(requests, 'post') as mock_post:
        mock_response = mock.Mock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response

        with mock.patch(
                'multi_modal_edge_ai.client.controllers.client_controller.send_set_up_connection_request'
        ) as mock_setup:
            send_heartbeat(client_config)

            mock_setup.assert_called_once()

            assert "Client not found" in caplog.text


def test_send_heartbeat_fail(client_config, caplog):
    with mock.patch.object(requests, 'post'):
        mock_response = mock.Mock()
        mock_response.status_code = 500

        # Call the method under test
        send_heartbeat(client_config, 0, 0)

        # Check the log
        assert "An error occurred during heartbeat with server: " in caplog.text


def test_save_both_models_zip_file(client_config):
    zip_file_path = 'tests/client/controllers/models_test_files/test_both_models.zip'
    with open(zip_file_path, 'rb') as file:
        zip_content = file.read()

    response = requests.Response()
    response._content = zip_content

    with patch('multi_modal_edge_ai.client.controllers.client_controller.save_model_file') as mock_save_model_file:
        save_models_zip_file(response, client_config['adl_model_keeper'],
                             client_config['anomaly_detection_model_keeper'])

        assert mock_save_model_file.call_count == 2


def test_save_adl_models_zip_file(client_config):
    zip_file_path = 'tests/client/controllers/models_test_files/test_adl_model.zip'
    with open(zip_file_path, 'rb') as file:
        zip_content = file.read()

    response = requests.Response()
    response._content = zip_content

    with patch('multi_modal_edge_ai.client.controllers.client_controller.save_model_file') as mock_save_model_file:
        save_models_zip_file(response, client_config['adl_model_keeper'],
                             client_config['anomaly_detection_model_keeper'])
        mock_save_model_file.assert_any_call(mock.ANY, client_config['adl_model_keeper'])
        assert mock_save_model_file.call_count == 1


def test_save_andet_models_zip_file(client_config):
    zip_file_path = 'tests/client/controllers/models_test_files/test_anomaly_detection_model.zip'
    with open(zip_file_path, 'rb') as file:
        zip_content = file.read()

    response = requests.Response()
    response._content = zip_content

    with patch('multi_modal_edge_ai.client.controllers.client_controller.save_model_file') as mock_save_model_file:
        save_models_zip_file(response, client_config['adl_model_keeper'],
                             client_config['anomaly_detection_model_keeper'])
        mock_save_model_file.assert_any_call(mock.ANY, client_config['anomaly_detection_model_keeper'])
        assert mock_save_model_file.call_count == 1


def test_save_no_models_zip_file(client_config):
    zip_file_path = 'tests/client/controllers/models_test_files/test_no_models.zip'
    with open(zip_file_path, 'rb') as file:
        zip_content = file.read()

    response = requests.Response()
    response._content = zip_content

    with patch('multi_modal_edge_ai.client.controllers.client_controller.save_model_file') as mock_save_model_file:
        save_models_zip_file(response, client_config['adl_model_keeper'],
                             client_config['anomaly_detection_model_keeper'])

        assert mock_save_model_file.call_count == 0


def test_save_model_file_adl(client_config):
    model_file = 'tests/client/controllers/models_test_files/test_model_source'
    model_path = client_config['adl_model_keeper'].model_path

    # Empty the destination file after the test
    with open(model_path, 'w') as file:
        file.truncate(0)

    with open(model_file, 'rb') as original_file:
        original_content = original_file.read()

    save_model_file(model_file, client_config['adl_model_keeper'])

    # Assert the saved file content matches the original file content
    with open(model_path, 'rb') as saved_file:
        saved_content = saved_file.read()

    assert saved_content == original_content, "Saved file content does not match the original file content"

    # Empty the destination file after the test
    with open(model_path, 'w') as file:
        file.truncate(0)


def test_save_model_file_invalid_path(client_config):
    model_file = 'invalid/path/to/model_file'

    with pytest.raises(Exception) as exc_info:
        save_model_file(model_file, client_config['adl_model_keeper'])

    assert str(exc_info.value) == "Error occurred while saving the model file:" + \
           " [Errno 2] No such file or directory: 'invalid/path/to/model_file'"
