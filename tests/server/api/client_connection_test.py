import io
import time
import zipfile
from datetime import datetime, timedelta
from multiprocessing import Process

import pytest

from multi_modal_edge_ai.server.main import app, get_connected_clients, update_anomaly_detection_model_update_time, \
    update_adl_model_update_time


@pytest.fixture
def client():
    with app.test_client() as client:
        client.environ_base['REMOTE_ADDR'] = '0.0.0.0'
        yield client


@pytest.fixture
def client1():
    with app.test_client() as client:
        client.environ_base['REMOTE_ADDR'] = '0.0.0.1'
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
    assert_connected_clients_with_expected(expected_data)


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
    assert_connected_clients_with_expected(expected_data)


def test_heartbeat_with_anomaly_detection_file(client):
    payload = {
        'recent_adls': 0,
        'recent_anomalies': 0
    }

    update_anomaly_detection_model_update_time(datetime.now() + timedelta(days=1))

    response = client.post('/api/heartbeat', json=payload)
    assert_response_with_zip(response, False, True)

    expected_data = {
        '0.0.0.0': {'status': 'Connected',
                    'num_adls': 5,
                    'num_anomalies': 5
                    }
    }
    assert_connected_clients_with_expected(expected_data)


def test_heartbeat_with_adl_file(client):
    payload = {
        'recent_adls': 0,
        'recent_anomalies': 0
    }

    update_anomaly_detection_model_update_time(datetime.now() - timedelta(days=2))
    update_adl_model_update_time(datetime.now() + timedelta(days=1))

    response = client.post('/api/heartbeat', json=payload)
    assert_response_with_zip(response, True, False)

    expected_data = {
        '0.0.0.0': {'status': 'Connected',
                    'num_adls': 5,
                    'num_anomalies': 5
                    }
    }
    assert_connected_clients_with_expected(expected_data)


def test_heartbeat_extra_adls(client):
    payload = {
        'recent_adls': 5,
        'recent_anomalies': 0
    }
    update_adl_model_update_time(datetime.now() - timedelta(days=2))

    response = client.post('api/heartbeat', json=payload)
    assert_response_with_zip(response, False, False)

    expected_data = {
        '0.0.0.0': {'status': 'Connected',
                    'num_adls': 10,
                    'num_anomalies': 5
                    }
    }
    assert_connected_clients_with_expected(expected_data)


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
    assert_connected_clients_with_expected(expected_data)


def test_heartbeat_unseen_client(client1):
    payload = {
        'recent_adls': 10,
        'recent_anomalies': 5
    }
    response = client1.post('api/heartbeat', json=payload)
    assert response.status_code == 404
    assert response.get_json() == {'message': 'Client not found'}

    expected_data = {
        '0.0.0.0': {'status': 'Connected',
                    'num_adls': 10,
                    'num_anomalies': 5
                    }
    }
    assert_connected_clients_with_expected(expected_data)


def assert_connected_clients_with_expected(expected):
    connected_clients = get_connected_clients()

    assert len(connected_clients) == 1
    ip = list(expected.keys())[0]
    assert connected_clients[ip]['status'] == expected[ip]['status']
    assert connected_clients[ip]['num_adls'] == expected[ip]['num_adls']
    assert connected_clients[ip]['num_anomalies'] == expected[ip]['num_anomalies']

    # Assert last_seen column is datetime
    assert isinstance(connected_clients[ip]['last_seen'], datetime)


def assert_response_with_zip(response, adl: bool, andet: bool):
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'application/zip'

    zip_content = io.BytesIO(response.data)
    with zipfile.ZipFile(zip_content, 'r') as zipfolder:
        zip_files = zipfolder.namelist()

        assert ('adl_model' in zip_files) == adl
        assert ('anomaly_detection_model' in zip_files) == andet
