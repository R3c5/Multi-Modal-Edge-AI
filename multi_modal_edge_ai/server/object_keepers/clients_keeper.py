from datetime import datetime, timedelta
from typing import Dict, Any, Optional


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
            'last_model_aggregation': datetime
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
                'last_model_aggregation': datetime.min,
                'num_adls': 0,
                'num_anomalies': 0
            }
            self.connected_clients[ip] = new_client

    def update_client(self, ip: str, status: str, last_seen: datetime, num_adls: int = 0,
                      num_anomalies: int = 0) -> None:
        """
        Update the client in the connected_clients with the same ip as the one passed as a parameter
        :param ip: string representing the ip of the new client
        :param status: string 'Connected' or 'Disconnected'
        :param last_seen: datetime of when the client last sent a message to the server
        :param num_adls: int representing the number of adls that will be added to this client
        :param num_anomalies: int representing the number of anomalies that will be added to this client
        """
        client = self.connected_clients[ip]
        client['status'] = status
        client['last_seen'] = last_seen
        client['num_adls'] += num_adls
        client['num_anomalies'] += num_anomalies

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

    def update_last_model_aggregation(self, ip: str, date: datetime) -> None:
        """
        This function will override the last_model_aggregation field
        :param ip: The ip of the client on which to override the last_model_aggregation field
        :param date: The datetime object to which override
        :return:
        """
        self.connected_clients[ip]["last_model_aggregation"] = date

    def get_last_seen(self, ip) -> Optional[datetime]:
        """
        Get the last seen of the client with the ip
        :param ip: client ip
        :return: datetime of the last seen
        """
        client = self.connected_clients.get(ip)
        if client is not None:
            return client['last_seen']
        return None

    def client_exists(self, ip) -> bool:
        """
        Return true iff ip is in connected clients
        :param ip: IP to check existence of
        """
        return ip in self.connected_clients
