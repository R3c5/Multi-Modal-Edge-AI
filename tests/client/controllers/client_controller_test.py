from unittest import mock
from unittest.mock import patch, mock_open

import pytest
import requests

from multi_modal_edge_ai.client.controllers.client_controller import send_set_up_connection_request, send_heartbeat, \
    save_models_zip_file, save_model_file


def test_send_set_up_connection_request_success(capsys):
    with mock.patch.object(requests, 'get') as mock_get:
        mock_response = mock.Mock()
        mock_response.status_code = 200

        with mock.patch('multi_modal_edge_ai.client.controllers.client_controller.save_models_zip_file') \
                as mock_save_zip:
            mock_get.return_value = mock_response
            # Call the method under test
            send_set_up_connection_request()
            # Assert that the appropriate functions were called
            mock_save_zip.assert_called_with(mock_get.return_value)

            captured = capsys.readouterr()
            assert 'Connection set up successfully' in captured.out


def test_send_set_up_connection_request_fail(capsys, caplog):
    with mock.patch.object(requests, 'get'):
        mock_response = mock.Mock()
        mock_response.status_code = 500
        mock_response.text = "Server error"

        send_set_up_connection_request()

        # Check the printed exception message
        captured = capsys.readouterr()
        assert 'Error setting up connection: ' in captured.out
        # Check the log
        assert "An error occurred during set up with server: " in caplog.text


def test_send_heartbeat_success(capsys):
    with mock.patch.object(requests, 'post') as mock_post:
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.headers = {'start_federation_client_flag': 'False'}

        with mock.patch('multi_modal_edge_ai.client.controllers.client_controller.save_models_zip_file') \
                as mock_save_zip:
            mock_post.return_value = mock_response
            # Call the method under test
            send_heartbeat(0, 0)
            # Assert that the appropriate functions were called
            mock_save_zip.assert_called_with(mock_post.return_value)

            captured = capsys.readouterr()
            assert 'Heartbeat successful\n' in captured.out


def test_send_heartbeat_no_client(capsys):
    with mock.patch.object(requests, 'post') as mock_post:
        mock_response = mock.Mock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response

        with mock.patch(
                'multi_modal_edge_ai.client.controllers.client_controller.send_set_up_connection_request'
        ) as mock_setup:
            send_heartbeat()

            mock_setup.assert_called_once()

            captured = capsys.readouterr()
            assert "Client not found" in captured.out


def test_send_heartbeat_fail(capsys, caplog):
    with mock.patch.object(requests, 'post'):
        mock_response = mock.Mock()
        mock_response.status_code = 500

        # Call the method under test
        send_heartbeat(0, 0)

        captured = capsys.readouterr()
        assert "Error sending heartbeat: " in captured.out
        # Check the log
        assert "An error occurred during heartbeat with server: " in caplog.text


def test_save_both_models_zip_file():
    zip_file_path = 'tests/client/controllers/models_test_files/test_both_models.zip'
    with open(zip_file_path, 'rb') as file:
        zip_content = file.read()

    response = requests.Response()
    response._content = zip_content

    with patch('multi_modal_edge_ai.client.controllers.client_controller.save_model_file') as mock_save_model_file:
        save_models_zip_file(response)

        assert mock_save_model_file.call_count == 2


def test_save_adl_models_zip_file():
    zip_file_path = 'tests/client/controllers/models_test_files/test_adl_model.zip'
    with open(zip_file_path, 'rb') as file:
        zip_content = file.read()

    response = requests.Response()
    response._content = zip_content

    with patch('multi_modal_edge_ai.client.controllers.client_controller.save_model_file') as mock_save_model_file:
        save_models_zip_file(response)
        mock_save_model_file.assert_any_call(mock.ANY, "ADL")
        assert mock_save_model_file.call_count == 1


def test_save_andet_models_zip_file():
    zip_file_path = 'tests/client/controllers/models_test_files/test_anomaly_detection_model.zip'
    with open(zip_file_path, 'rb') as file:
        zip_content = file.read()

    response = requests.Response()
    response._content = zip_content

    with patch('multi_modal_edge_ai.client.controllers.client_controller.save_model_file') as mock_save_model_file:
        save_models_zip_file(response)
        mock_save_model_file.assert_any_call(mock.ANY, "AnDet")
        assert mock_save_model_file.call_count == 1


def test_save_no_models_zip_file():
    zip_file_path = 'tests/client/controllers/models_test_files/test_no_models.zip'
    with open(zip_file_path, 'rb') as file:
        zip_content = file.read()

    response = requests.Response()
    response._content = zip_content

    with patch('multi_modal_edge_ai.client.controllers.client_controller.save_model_file') as mock_save_model_file:
        save_models_zip_file(response)

        assert mock_save_model_file.call_count == 0


def test_save_model_file_adl():
    model_file = 'tests/client/controllers/models_test_files/test_model_source'
    keeper_type = 'ADL'
    adl_model_keeper_path = 'tests/client/controllers/models_test_files/test_model_dest'

    # Empty the destination file after the test
    with open(adl_model_keeper_path, 'w') as file:
        file.truncate(0)

    with open(model_file, 'rb') as original_file:
        original_content = original_file.read()

    with patch('multi_modal_edge_ai.client.main.adl_model_keeper') as mock_model_keeper:
        mock_model_keeper.model_path = adl_model_keeper_path
        save_model_file(model_file, keeper_type)

        # Assert the saved file content matches the original file content
        with open(adl_model_keeper_path, 'rb') as saved_file:
            saved_content = saved_file.read()

        assert saved_content == original_content, "Saved file content does not match the original file content"

        # Empty the destination file after the test
        with open(adl_model_keeper_path, 'w') as file:
            file.truncate(0)


def test_save_model_file_andet():
    model_file = 'tests/client/controllers/models_test_files/test_model_source'
    keeper_type = 'AnDet'
    anomaly_det_model_keeper_path = 'tests/client/controllers/models_test_files/test_model_dest'

    # Empty the destination file after the test
    with open(anomaly_det_model_keeper_path, 'w') as file:
        file.truncate(0)

    with open(model_file, 'rb') as original_file:
        original_content = original_file.read()

    with patch('multi_modal_edge_ai.client.main.anomaly_detection_model_keeper') as mock_model_keeper:
        mock_model_keeper.model_path = anomaly_det_model_keeper_path
        save_model_file(model_file, keeper_type)

        # Assert the saved file content matches the original file content
        with open(anomaly_det_model_keeper_path, 'rb') as saved_file:
            saved_content = saved_file.read()

        assert saved_content == original_content, "Saved file content does not match the original file content"

        # Empty the destination file after the test
        with open(anomaly_det_model_keeper_path, 'w') as file:
            file.truncate(0)


def test_save_model_file_invalid_keeper_type():
    model_file = 'adl_test_model'
    invalid_keeper_type = 'InvalidKeeperType'

    with patch('multi_modal_edge_ai.client.main.adl_model_keeper'), \
            patch('multi_modal_edge_ai.client.main.anomaly_detection_model_keeper'):

        try:
            save_model_file(model_file, invalid_keeper_type)
            assert False, "Expected an exception to be raised for invalid keeper_type"
        except Exception as e:
            assert str(e) == "Expected keeper_type to be either ADL or AnDet!", \
                "Exception message does not match the expected value"


def test_save_model_file_invalid_path():
    model_file = 'invalid/path/to/model_file'
    keeper_type = 'ADL'

    with pytest.raises(Exception) as exc_info:
        save_model_file(model_file, keeper_type)

    assert str(exc_info.value) == "Error occurred while saving the model file:" +\
                                  " [Errno 2] No such file or directory: 'invalid/path/to/model_file'"
