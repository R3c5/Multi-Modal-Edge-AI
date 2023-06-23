import logging
import os
import threading
import time
from unittest.mock import patch

import pytest
from flask import Flask
from torch import nn

from multi_modal_edge_ai.client.common.adl_model_keeper import ADLModelKeeper
from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper
from multi_modal_edge_ai.client.controllers.client_controller import send_set_up_connection_request, send_heartbeat
from multi_modal_edge_ai.commons.string_label_encoder import StringLabelEncoder
from multi_modal_edge_ai.models.adl_inference.ml_models.svm_model import SVMModel
from multi_modal_edge_ai.models.anomaly_detection.ml_models import Autoencoder
from multi_modal_edge_ai.server.main import run_server_set_up


@pytest.fixture(autouse=True)
def set_logging_level(caplog):
    caplog.set_level(logging.INFO)


def run_server(app):
    with app.app_context():
        run_server_set_up(app)
    # Check the stop condition before exiting the thread
    if stop_server:
        return


@pytest.fixture(scope="module")
def run_server_fixture():
    app = Flask(__name__)

    global stop_server
    stop_server = False

    server_thread = threading.Thread(target=run_server, args=(app,), daemon=True)
    server_thread.start()
    time.sleep(3)

    yield app

    # Set the stop condition to True to signal the thread to exit
    stop_server = True
    server_thread.join(timeout=1)


@pytest.fixture(scope="function")
def app(run_server_fixture):
    yield run_server_fixture


@pytest.fixture(scope="function")
def client(app, run_server_fixture):
    # Create a test client using the Flask app
    with app.test_client() as client:
        yield client


@pytest.fixture(scope="module")
def client_config():
    root_directory = os.path.abspath(os.path.dirname(__file__))
    model_data_directory = os.path.join(root_directory, 'model_data')

    distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside',
                         'Movement']
    adl_encoder = StringLabelEncoder(distinct_adl_list)
    adl_model = SVMModel()
    adl_model_path = os.path.join(model_data_directory, 'adl_model')
    adl_encoder_path = os.path.join(model_data_directory, 'adl_encoder')
    adl_model_keeper = ADLModelKeeper(adl_model, adl_model_path, adl_encoder, adl_encoder_path)

    anomaly_detection_model = Autoencoder([96, 64, 32, 24, 16, 8], [8, 16, 24, 32, 64, 96], nn.ReLU(), nn.Sigmoid())
    anomaly_detection_model_path = os.path.join(model_data_directory, 'adl_model')
    anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_detection_model_path)

    return {
        'adl_model_keeper': adl_model_keeper,
        'anomaly_detection_model_keeper': anomaly_detection_model_keeper
    }


def test_heartbeat_no_setup(client_config, client, caplog):
    with patch('multi_modal_edge_ai.client.controllers.client_controller.send_set_up_connection_request') as mock_setup:
        send_heartbeat(client_config, 1, 1)
        mock_setup.assert_called_once()

        assert 'Client not found' in caplog.text


def test_set_up_connection(client_config, client, caplog):
    with patch('multi_modal_edge_ai.client.controllers.client_controller.save_models_zip_file') as mock_save_zip:
        send_set_up_connection_request(client_config['adl_model_keeper'],
                                       client_config['anomaly_detection_model_keeper'])
        mock_save_zip.assert_called_once()

        assert 'Connection set up successfully' in caplog.text


def test_heartbeat(client_config, client, caplog):
    with patch('multi_modal_edge_ai.client.controllers.client_controller.save_models_zip_file') as mock_save_zip:
        send_set_up_connection_request(client_config['adl_model_keeper'],
                                       client_config['anomaly_detection_model_keeper'])
        send_heartbeat(client_config, 1, 1)
        mock_save_zip.assert_called()
        assert mock_save_zip.call_count == 2
        assert 'Heartbeat successful' in caplog.text