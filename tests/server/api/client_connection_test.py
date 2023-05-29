import pytest
from datetime import datetime

from multi_modal_edge_ai.server.api.client_connection import get_connected_clients_dict
from multi_modal_edge_ai.server.main import app


@pytest.fixture
def client():
    with app.test_client() as client:
        client.environ_base['REMOTE_ADDR'] = '0.0.0.0'
        yield client

def test_set_up_connection(client):
    response = client.get('/api/set_up_connection')
    assert response.status_code == 200
    assert response.get_json() == {'message': 'Connection set up successfully'}

    # Assert connected_clients dictionary
    connected_clients = get_connected_clients_dict()
    assert len(connected_clients) == 1
    client_ip = list(connected_clients.keys())[0]
    timestamp = connected_clients[client_ip]
    assert client_ip == '0.0.0.0'
    assert isinstance(timestamp, datetime)


def test_heartbeat(client):
    # seen client
    client.get('/api/set_up_connection')
    response = client.post('api/heartbeat')
    assert response.status_code == 200
    assert response.get_json() == {'message': 'Heartbeat received'}

    # Assert connected_clients dictionary
    connected_clients = get_connected_clients_dict()
    assert len(connected_clients) == 1
    client_ip = list(connected_clients.keys())[0]
    timestamp = connected_clients[client_ip]
    assert client_ip == '0.0.0.0'
    assert isinstance(timestamp, datetime)


    # unseen client
    with app.test_client() as unseen_client:
        unseen_client.environ_base['REMOTE_ADDR'] = '0.0.0.1'
    response = unseen_client.post('/api/heartbeat')
    assert response.status_code == 404
    assert response.get_json() == {'message': 'Client not found'}

    # Assert connected_clients dictionary remains unchanged
    connected_clients = get_connected_clients_dict()
    assert len(connected_clients) == 1
    client_ip = list(connected_clients.keys())[0]
    timestamp = connected_clients[client_ip]
    assert client_ip == '0.0.0.0'
    assert isinstance(timestamp, datetime)
