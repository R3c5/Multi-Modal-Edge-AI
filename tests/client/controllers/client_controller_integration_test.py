import time
from multiprocessing import Process
from unittest.mock import patch

import pytest

from multi_modal_edge_ai.client.controllers.client_controller import send_set_up_connection_request, send_heartbeat
from multi_modal_edge_ai.server.main import app


@pytest.fixture(scope="module")
def client():
    # Create a test client using the Flask app
    with app.test_client() as client:
        yield client


def start_server():
    app.run(port=5000)


@pytest.fixture(scope="function", autouse=True)
def run_server():
    server_process = Process(target=start_server)
    server_process.start()

    time.sleep(2)

    yield

    server_process.terminate()
    server_process.join()


def test_set_up_connection(client, capsys):
    with patch('multi_modal_edge_ai.client.controllers.client_controller.save_models_zip_file') as mock_save_zip:
        send_set_up_connection_request()
        mock_save_zip.assert_called_once()

        captured = capsys.readouterr()
        assert 'Connection set up successfully' in captured.out


def test_heartbeat_no_setup(client, capsys):
    with patch('multi_modal_edge_ai.client.controllers.client_controller.send_set_up_connection_request') as mock_setup:
        send_heartbeat(1, 1)
        mock_setup.assert_called_once()

        captured = capsys.readouterr()
        assert 'Client not found' in captured.out


def test_heartbeat(client, capsys):
    with patch('multi_modal_edge_ai.client.controllers.client_controller.save_models_zip_file') as mock_save_zip:
        send_set_up_connection_request()
        send_heartbeat(1, 1)
        mock_save_zip.assert_called()
        assert mock_save_zip.call_count == 2
        captured = capsys.readouterr()
        assert 'Heartbeat successful' in captured.out
