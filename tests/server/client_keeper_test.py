import datetime
import pandas as pd
from multi_modal_edge_ai.server.clients_keeper import ClientsKeeper


def test_add_client():
    keeper = ClientsKeeper()
    ip = '192.168.1.1'
    status = 'Connected'
    last_seen = datetime.datetime.now()

    keeper.add_client(ip, status, last_seen)

    # Assert that the client is added correctly
    assert len(keeper.connected_clients) == 1
    assert keeper.connected_clients.loc[0, 'ip'] == ip
    assert keeper.connected_clients.loc[0, 'status'] == status
    assert keeper.connected_clients.loc[0, 'last_seen'] == last_seen
    assert keeper.connected_clients.loc[0, 'num_adls'] == 0
    assert keeper.connected_clients.loc[0, 'num_anomalies'] == 0


def test_update_client():
    keeper = ClientsKeeper()
    ip = '192.168.1.1'
    status = 'Connected'
    last_seen = datetime.datetime.now()
    num_adls = 2
    num_anomalies = 1

    keeper.add_client(ip, status, last_seen)
    keeper.update_client(ip, status, last_seen, num_adls, num_anomalies)

    # Assert that the client is updated correctly
    assert keeper.connected_clients.loc[0, 'num_adls'] == num_adls
    assert keeper.connected_clients.loc[0, 'num_anomalies'] == num_anomalies


def test_update_clients_statuses():
    keeper = ClientsKeeper()
    ip1 = '192.168.1.1'
    ip2 = '192.168.1.2'
    current_time = datetime.datetime.now()

    # Add two clients with different last_seen times
    keeper.add_client(ip1, 'Connected', current_time - datetime.timedelta(hours=2))
    keeper.add_client(ip2, 'Connected', current_time - datetime.timedelta(hours=4))

    # Update clients statuses
    keeper.update_clients_statuses()

    # Assert that the statuses are updated correctly
    assert keeper.connected_clients.loc[0, 'status'] == 'Connected'
    assert keeper.connected_clients.loc[1, 'status'] == 'Disconnected'
