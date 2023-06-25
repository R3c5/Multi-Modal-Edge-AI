import threading
import time

import pytest
from flask import Flask

from multi_modal_edge_ai.server.main import run_server_set_up


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
    with app.test_client() as client:
        client.environ_base['REMOTE_ADDR'] = '0.0.0.0'
        yield client


def test_get_client_info(client):
    client.get('/api/set_up_connection')
    payload = {
        'recent_adls': 10,
        'recent_anomalies': 5
    }
    client.post('api/heartbeat', json=payload)

    # Set up the headers with the authorization token
    file = open("./multi_modal_edge_ai/server/developer_dashboard/token.txt", 'r')
    token = file.read().strip()
    headers = {'Authorization': token}

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

    for ip, expected_values in expected.items():
        assert ip in connected_clients
        connected_values = connected_clients[ip]
        for key, value in expected_values.items():
            assert key in connected_values
            assert connected_values[key] == value


@pytest.fixture(autouse=True, scope="module")
def teardown_server_session(request):
    def stop_server_thread():
        global stop_server
        stop_server = True

    # Add the stop_server_thread function to the teardown callbacks
    request.addfinalizer(stop_server_thread)
