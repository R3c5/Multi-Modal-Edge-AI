import time
from multiprocessing import Process

import pytest

from multi_modal_edge_ai.server.main import app, run_server_set_up
from tests.server.api.client_connection_test import assert_connected_clients_with_expected


def startup_server():
    run_server_set_up()


@pytest.fixture(scope="function", autouse=True)
def run_server():
    server_process = Process(target=startup_server)
    server_process.start()

    time.sleep(3)

    yield

    server_process.terminate()
    server_process.join()


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_get_client_info(client):
    # Set up the headers with the authorization token
    headers = {'Authorization': 'super_secure_token_here_123'}

    # Make a GET request to the API endpoint
    response = client.get('/dashboard/get_client_info', headers=headers)

    # Assert the response status code is 200 (OK)
    assert response.status_code == 200

    # Assert the response JSON data contains the expected keys
    data = response.get_json()
    assert 'connected_clients' in data
    assert isinstance(data['connected_clients'], dict)

    # You can add more assertions to validate the response data
    connected_clients = data['connected_clients']

    assert len(connected_clients) == 1
    expected = {
        '0.0.0.0': {'status': 'Connected',
                    'num_adls': 10,
                    'num_anomalies': 5
                    }
    }

    assert_connected_clients_with_expected(expected)
