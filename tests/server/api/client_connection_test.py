import logging
from unittest.mock import patch

import pandas as pd
import pytest
from datetime import datetime

from pandas._testing import assert_frame_equal

from multi_modal_edge_ai.server.main import app, get_connected_clients


@pytest.fixture
def client():
    with app.test_client() as client:
        client.environ_base['REMOTE_ADDR'] = '0.0.0.0'
        yield client


def test_set_up_connection(client):
    response = client.get('/api/set_up_connection')
    assert response.status_code == 200
    assert response.get_json() == {'message': 'Connection set up successfully'}

    # Create expected DataFrame
    expected_data = {
        'ip': ['0.0.0.0'],
        'status': ['Connected'],
        'num_adls': [0],
        'num_anomalies': [0]
    }
    expected_df = pd.DataFrame(expected_data)
    assert_connected_clients_with_expected(expected_df)

    # def test_heartbeat_seen_client(client):
    #     client.get('/api/set_up_connection')
    #     response = client.post('api/heartbeat')
    #     assert response.status_code == 200
    #     assert response.get_json() == {'message': 'Heartbeat received'}
    #
    #     # Assert connected_clients dictionary
    #     connected_clients = get_connected_clients()
    #     assert len(connected_clients) == 1
    #     client_ip = list(connected_clients.keys())[0]
    #     timestamp = connected_clients[client_ip]
    #     assert client_ip == '0.0.0.0'
    #     assert isinstance(timestamp, datetime)
    #
    #

# def test_heartbeat_unseen_client():
#     with app.test_client() as unseen_client:
#         unseen_client.environ_base['REMOTE_ADDR'] = '0.0.0.1'
#     response = unseen_client.post('/api/heartbeat')
#     assert response.status_code == 404
#     assert response.get_json() == {'message': 'Client not found'}
#
#     # Create expected DataFrame
#     expected_data = {
#         'ip': ['0.0.0.0'],
#         'status': ['Connected'],
#         'num_adls': [0],
#         'num_anomalies': [0]
#     }
#     assert_connected_clients_with_expected(pd.DataFrame(expected_data))


def assert_connected_clients_with_expected(expected):
    connected_clients = get_connected_clients()
    connected_clients_without_last_seen = connected_clients.drop('last_seen', axis=1)
    assert_frame_equal(connected_clients_without_last_seen, expected)

    # Assert last_seen column is datetime
    assert isinstance(connected_clients['last_seen'].iloc[0], datetime)
