from datetime import datetime
from unittest.mock import patch

import pytest
from multi_modal_edge_ai.server.main import app


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
    assert isinstance(data['connected_clients'], list)

    # You can add more assertions to validate the response data
    connected_clients = data['connected_clients']
    assert len(connected_clients) == 1
    client_ip = connected_clients[0]['ip']
    assert client_ip == '0.0.0.0'
