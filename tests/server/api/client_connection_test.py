import io
import zipfile

import pytest
from datetime import datetime

from multi_modal_edge_ai.server.main import app, get_connected_clients


@pytest.fixture
def client():
    with app.test_client() as client:
        client.environ_base['REMOTE_ADDR'] = '0.0.0.0'
        yield client


def test_set_up_connection(client):
    response = client.get('/api/set_up_connection')
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'application/zip'

    with zipfile.ZipFile(io.BytesIO(response.data), 'r') as zipfolder:
        # Check if the ADL model file exists in the ZIP
        assert 'adl_model' in zipfolder.namelist()

        # Check if the anomaly detection model file exists in the ZIP
        assert 'anomaly_detection_model' in zipfolder.namelist()

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
    assert response.status_code == 200
    assert response.get_json() == {'message': 'No new model updates'}

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
    response = client.post('api/heartbeat', json=payload)
    assert response.status_code == 200
    assert response.get_json() == {'message': 'No new model updates'}

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


def test_heartbeat_unseen_client():
    with app.test_client() as unseen_client:
        unseen_client.environ_base['REMOTE_ADDR'] = '0.0.0.1'
    payload = {
        'recent_adls': 10,
        'recent_anomalies': 5
    }
    response = unseen_client.post('api/heartbeat', json=payload)
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
