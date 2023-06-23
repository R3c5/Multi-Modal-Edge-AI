import threading
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
            'start_federation': bool,
            'start_personalization': bool
            'num_adls': int,
            'num_anomalies': int
            'last_model_aggregation': datetime,
            'last_model_personalization': datetime
        },
        ...
    }
    """

    def __init__(self) -> None:
        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.daily_information_lock = threading.Lock()
        self.start_federation_lock = threading.Lock()

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
                'last_model_personalization': datetime.min,
                'start_federation': False,
                'start_personalization': False,
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
        with self.daily_information_lock:
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

    def update_last_model_personalization(self, ip: str, date: datetime) -> None:
        """
        This function will override the last_model_personalization field
        :param ip: The ip of the client on which to override the last_model_personalization field
        :param date: The datetime object to which override
        :return:
        """
        self.connected_clients[ip]["last_model_personalization"] = date

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

    def reset_all_daily_information(self) -> None:
        """
        This function resets the values of the num_adls and num_anomalies for all the clients.
        :return:
        """
        with self.daily_information_lock:
            for client in self.connected_clients.keys():
                self.connected_clients[client]["num_adls"] = 0
                self.connected_clients[client]["num_anomalies"] = 0

    def set_start_workload(self, workload_flag: str, value: bool) -> None:
        """
        This function will set the flag of the specified type of workload as specified. It will do so for all the
        clients
        :param workload_flag: The type of workload: start_federation or start_personalization
        :param value: The value, true or false
        :return:
        """
        with self.start_federation_lock:
            for ip in self.connected_clients.keys():
                self.connected_clients[ip][workload_flag] = value

    def compare_and_swap_start_workload(self, workload_flag: str, client_ip: str) -> bool:
        """
        This function will compare and swap the value of the flag for the workload. This is an atomic operation, and if
        the value of the flag is true, it will set it to false. If it is false, it will remain false
        :param workload_flag: The type of workload: start_federation or start_personalization
        :param client_ip: The ip of the client for which to perform the compare and swap
        :return: The value after swap
        """
        with self.start_federation_lock:
            if not self.connected_clients[client_ip][workload_flag]:
                return False
            else:
                self.connected_clients[client_ip][workload_flag] = False
                return True
