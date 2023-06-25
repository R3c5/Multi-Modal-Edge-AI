import datetime

from multi_modal_edge_ai.server.object_keepers.clients_keeper import ClientsKeeper


def test_add_client():
    keeper = ClientsKeeper()
    ip = '192.168.1.1'
    status = 'Connected'
    last_seen = datetime.datetime.now()

    keeper.add_client(ip, status, last_seen)

    # Assert that the client is added correctly
    assert len(keeper.connected_clients) == 1
    assert keeper.connected_clients[ip]['status'] == status
    assert keeper.connected_clients[ip]['last_seen'] == last_seen
    assert keeper.connected_clients[ip]['num_adls'] == 0
    assert keeper.connected_clients[ip]['num_anomalies'] == 0


def test_add_old_client():
    keeper = ClientsKeeper()
    ip = '192.168.1.1'
    status = 'Connected'
    last_seen = datetime.datetime.now()

    keeper.add_client(ip, status, last_seen)
    keeper.add_client(ip, 'Disconnected', last_seen)

    # Assert that the client is added correctly
    assert len(keeper.connected_clients) == 1
    assert keeper.connected_clients[ip]['status'] == 'Disconnected'
    assert keeper.connected_clients[ip]['last_seen'] == last_seen
    assert keeper.connected_clients[ip]['num_adls'] == 0
    assert keeper.connected_clients[ip]['num_anomalies'] == 0


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
    assert keeper.connected_clients[ip]['num_adls'] == num_adls
    assert keeper.connected_clients[ip]['num_anomalies'] == num_anomalies


def test_update_clients_statuses():
    keeper = ClientsKeeper()
    ip1 = '192.168.1.1'
    ip2 = '192.168.1.2'
    current_time = datetime.datetime.now()

    # Add two clients with different last_seen times
    keeper.add_client(ip1, 'Connected', current_time - datetime.timedelta(seconds=20))
    keeper.add_client(ip2, 'Connected', current_time - datetime.timedelta(seconds=40))

    # Update clients statuses
    keeper.update_clients_statuses()

    # Assert that the statuses are updated correctly
    assert keeper.connected_clients[ip1]['status'] == 'Connected'
    assert keeper.connected_clients[ip2]['status'] == 'Disconnected'


def test_get_last_seen():
    keeper = ClientsKeeper()

    last_seen = datetime.datetime.now() - datetime.timedelta(hours=2)
    keeper.add_client('192.168.0.1', 'Connected', last_seen)
    keeper.add_client('192.168.0.2', 'Connected', datetime.datetime.now())

    result = keeper.get_last_seen('192.168.0.1')

    assert last_seen == result


def test_get_last_seen_none():
    keeper = ClientsKeeper()

    last_seen = datetime.datetime.now() - datetime.timedelta(hours=2)
    keeper.add_client('192.168.0.1', 'Connected', last_seen)
    keeper.add_client('192.168.0.2', 'Connected', datetime.datetime.now())

    result = keeper.get_last_seen('192.168.0.3')

    assert result is None


def test_exists_client():
    keeper = ClientsKeeper()

    keeper.add_client('192.168.0.1', 'Connected', datetime.datetime.now())
    keeper.add_client('192.168.0.2', 'Connected', datetime.datetime.now())

    assert keeper.client_exists('192.168.0.2')
    assert not keeper.client_exists('192.168.0.3')


def test_reset_all_daily_information():
    keeper = ClientsKeeper()

    keeper.add_client('192.168.0.1', 'Connected', datetime.datetime.now())
    keeper.add_client('192.168.0.2', 'Connected', datetime.datetime.now())

    keeper.update_client('192.168.0.1', 'Connected', datetime.datetime.now(), 2, 3)
    keeper.update_client('192.168.0.2', 'Connected', datetime.datetime.now(), 2, 3)

    keeper.reset_all_daily_information()

    assert keeper.connected_clients["192.168.0.1"]["num_adls"] == 0
    assert keeper.connected_clients["192.168.0.2"]["num_adls"] == 0
    assert keeper.connected_clients["192.168.0.1"]["num_anomalies"] == 0
    assert keeper.connected_clients["192.168.0.2"]["num_anomalies"] == 0
