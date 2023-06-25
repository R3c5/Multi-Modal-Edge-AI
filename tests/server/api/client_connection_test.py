import io
import multiprocessing
import threading
import time
import zipfile
from datetime import datetime
from multiprocessing import Process

import pytest
from flask import Flask

from multi_modal_edge_ai.server.api.client_connection import client_connection_blueprint
from multi_modal_edge_ai.server.api.dashboard_connection import dashboard_connection_blueprint
from multi_modal_edge_ai.server.main import run_server_set_up, initialize_models, initialize_clients_keeper, \
    configure_client_connection_blueprints, configure_dashboard_connection_blueprints


def run_server(app):
    with app.app_context():
        run_server_set_up(app)
    # Check the stop condition before exiting the thread
    if stop_server:
        return


# @pytest.fixture(scope="session")
# def close_pytest():
#     pytest.exit()


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
    with app.test_client() as client:
        client.environ_base['REMOTE_ADDR'] = '0.0.0.0'
        yield client


@pytest.fixture(scope="function")
def client1(app, run_server_fixture):
    with app.test_client() as client:
        client.environ_base['REMOTE_ADDR'] = '0.0.0.1'
        yield client


def test_set_up_connection(client):
    response = client.get('/api/set_up_connection')
    assert_response_with_zip(response, True, True)

    expected_data = {
        '0.0.0.0': {
            'status': 'Connected',
            'num_adls': 0,
            'num_anomalies': 0
        }
    }
    assert_connected_clients_with_expected(client, expected_data)


def test_heartbeat_seen_client(client):
    payload = {
        'recent_adls': 5,
        'recent_anomalies': 5
    }
    response = client.post('api/heartbeat', json=payload)
    assert_response_with_zip(response, False, False)

    expected_data = {
        '0.0.0.0': {'status': 'Connected',
                    'num_adls': 5,
                    'num_anomalies': 5
                    }
    }
    assert_connected_clients_with_expected(client, expected_data)


def test_heartbeat_unseen_client(client1):
    payload = {
        'recent_adls': 10,
        'recent_anomalies': 5
    }
    response = client1.post('/api/heartbeat', json=payload)
    assert response.status_code == 404
    assert response.get_json() == {'message': 'Client not found'}

    file = open("./multi_modal_edge_ai/server/developer_dashboard/token.txt", 'r')
    token = file.read().strip()
    headers = {'Authorization': token}
    response = client1.get('/dashboard/get_client_info', headers=headers)
    print(response)
    connected_clients = response.get_json()['connected_clients']
    assert '0.0.0.1' not in list(connected_clients.keys())


def test_heartbeat_extra_adls(client):
    payload = {
        'recent_adls': 5,
        'recent_anomalies': 0
    }
    # update_adl_model_update_time(datetime.now() - timedelta(days=2))

    response = client.post('api/heartbeat', json=payload)
    assert_response_with_zip(response, False, False)

    expected_data = {
        '0.0.0.0': {'status': 'Connected',
                    'num_adls': 10,
                    'num_anomalies': 5
                    }
    }
    assert_connected_clients_with_expected(client, expected_data)


def test_heartbeat_bad_payload(client):
    payload = {
        'other_data': 42
    }
    response = client.post('api/heartbeat', json=payload)
    assert response.status_code == 400
    assert response.get_json() == {'message': 'Invalid JSON payload'}

    expected_data = {
        '0.0.0.0': {'status': 'Connected',
                    'num_adls': 10,
                    'num_anomalies': 5
                    }
    }
    assert_connected_clients_with_expected(client, expected_data)


def assert_connected_clients_with_expected(client, expected):
    file = open("./multi_modal_edge_ai/server/developer_dashboard/token.txt", 'r')
    token = file.read().strip()
    headers = {'Authorization': token}
    response = client.get('/dashboard/get_client_info', headers=headers)

    connected_clients = response.get_json()['connected_clients']

    assert len(connected_clients) == 1
    ip = list(expected.keys())[0]
    assert connected_clients[ip]['status'] == expected[ip]['status']
    assert connected_clients[ip]['num_adls'] == expected[ip]['num_adls']
    assert connected_clients[ip]['num_anomalies'] == expected[ip]['num_anomalies']

    # Assert last_seen column is datetime
    last_seen_str = connected_clients[ip]['last_seen']
    last_seen = datetime.strptime(last_seen_str, '%a, %d %b %Y %H:%M:%S %Z')
    assert isinstance(last_seen, datetime)


def assert_response_with_zip(response, adl: bool, andet: bool):
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'application/zip'

    zip_content = io.BytesIO(response.data)
    with zipfile.ZipFile(zip_content, 'r') as zipfolder:
        zip_files = zipfolder.namelist()

        assert ('adl_model' in zip_files) == adl
        assert ('anomaly_detection_model' in zip_files) == andet
