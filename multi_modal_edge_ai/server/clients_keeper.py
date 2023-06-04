from datetime import datetime, timedelta
from typing import Dict, Any


class ClientsKeeper:
    """
    This class holds all the information of the clients connected
    {connected_clients} has the following structure:
    {
        'ip1': {
            'status': 'Connected' or 'Disconnected',
            'last_seen': datetime,
            'num_adls': int,
            'num_anomalies': int
        },
        ...
    }
    """

    def __init__(self) -> None:
        self.connected_clients: Dict[str, Dict[str, Any]] = {}

    def add_client(self, ip: str, status: str, last_seen: datetime) -> None:
        """
        Add a new client to the connected_clients dictionary. If ip already exists, last_seen is updated.
        :param ip: string representing the ip of the new client
        :param status: string 'Connected' or 'Disconnected'
        :param last_seen: datetime of when the client last sent a message to the server
        """
        if ip in self.connected_clients:
            self.update_client(ip, status, last_seen)
        else:
            new_client = {
                'status': status,
                'last_seen': last_seen,
                'num_adls': 0,
                'num_anomalies': 0
            }
            self.connected_clients[ip] = new_client

    def update_client(self, ip: str, status: str, last_seen: datetime, num_adls: int = 0,
                      num_anomalies: int = 0) -> bool:
        """
        Update the client in the connected_clients with the same ip as the one passed as a parameter
        :param ip: string representing the ip of the new client
        :param status: string 'Connected' or 'Disconnected'
        :param last_seen: datetime of when the client last sent a message to the server
        :param num_adls: int representing the number of adls that will be added to this client
        :param num_anomalies: int representing the number of anomalies that will be added to this client
        :return True if update was successful and False if no client with the expected ip was found
        """
        if ip not in self.connected_clients:
            return False

        client = self.connected_clients[ip]
        client['status'] = status
        client['last_seen'] = last_seen
        client['num_adls'] += num_adls
        client['num_anomalies'] += num_anomalies

        return True

    def update_clients_statuses(self) -> None:
        """
        Update the statuses of the clients that were last seen more than 3 hours ago to 'Disconnected'
        """
        current_time = datetime.now()
        timeout_threshold = timedelta(hours=3)

        for ip, client in self.connected_clients.items():
            last_seen = client['last_seen']
            if (current_time - last_seen) > timeout_threshold:
                client['status'] = 'Disconnected'
            else:
                client['status'] = 'Connected'
