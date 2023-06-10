from unittest import mock
from unittest.mock import patch

import requests

from multi_modal_edge_ai.client.controllers.client_controller import send_set_up_connection_request, send_heartbeat, \
    save_model_file


def test_send_set_up_connection_request_success():
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

        with mock.patch('multi_modal_edge_ai.client.controllers.client_controller.save_models_zip_file') \
                as mock_save_zip:
            mock_post.return_value = mock_response
            # Call the method under test
            send_heartbeat(0, 0)
            # Assert that the appropriate functions were called
            mock_save_zip.assert_called_with(mock_post.return_value)

            captured = capsys.readouterr()
            assert 'Heartbeat successful\n' == captured.out


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
