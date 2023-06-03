from datetime import datetime, timedelta
from typing import Dict, Any, Union

import pandas as pd


class ClientsKeeper:
    """
    This class holds all the information of the clients connected
    {connected_clients} has the following column: ip, status, last_seen, num_adls, num_anomalies
    * ip represents the ip of the client
    * status is a string, either 'Connected' or 'Disconnected'
    * last_seen is a datetime representing the last time the client sent a request to the server
    * num_adls is the number of adls the client has predicted in certain period of time
    * num_anomalies is the number of anomalies the client has predicted in certain period of time
    """
    def __init__(self) -> None:
        self.client_columns = ['ip', 'status', 'last_seen', 'num_adls', 'num_anomalies']
        self.column_types: Dict[str, Any] = {'ip': str, 'status': str, 'last_seen': 'datetime64[s]', 'num_adls': int,
                                             'num_anomalies': int}
        df = pd.DataFrame(columns=self.client_columns)
        self.connected_clients = df.astype(self.column_types)

    def add_client(self, ip: str, status: str, last_seen: datetime) -> None:
        """
        Add a new client to the connected_clients dataframe. If ip already exists in df, last_seen is updated.
        :param ip: string representing the ip of the new client
        :param status: string 'Connected', or 'Disconnected'
        :param last_seen: datetime of when the client last sent a message to the server
        """
        if ip in self.connected_clients['ip'].values:
            self.update_client(ip, status, last_seen)
        else:
            new_client = {
                'ip': ip,
                'status': status,
                'last_seen': last_seen,
                'num_adls': 0,
                'num_anomalies': 0
            }
            self.connected_clients = pd.concat([self.connected_clients,
                                                pd.DataFrame([new_client], columns=self.client_columns)],
                                               ignore_index=True)

    def update_client(self, ip: str, status: str, last_seen: datetime, num_adls: Union[None, int] = None,
                      num_anomalies: Union[None, int] = None) -> bool:
        """
        Update the client in the connected_clients with the same ip as the one passed as a parameter
        :param ip: string representing the ip of the new client
        :param status: string 'Connected', or 'Disconnected'
        :param last_seen: datetime of when the client last sent a message to the server
        :param num_adls: int representing the number of adls that will be added to this client
        :param num_anomalies: int representing the number of anomalies that will be added to this client
        :return True if update was successful and False if no client with the expected ip was found
        """

        if ip not in self.connected_clients['ip'].values:
            return False
        index = self.connected_clients.index[self.connected_clients['ip'] == ip].tolist()[0]

        # Update the values in the row
        self.connected_clients.loc[index, 'status'] = status
        self.connected_clients.loc[index, 'last_seen'] = last_seen
        if num_adls is not None:
            self.connected_clients.loc[index, 'num_adls'] += num_adls
        if num_anomalies is not None:
            self.connected_clients.loc[index, 'num_anomalies'] += num_anomalies
        return True

    def update_clients_statuses(self) -> None:
        """
        update the statuses of the clients that were last seen more than 3 hours ago to 'Disconnected'
        """
        current_time = datetime.now()
        timeout_threshold = timedelta(hours=3)

        self.connected_clients['status'] = self.connected_clients.apply(
            lambda row: 'Disconnected' if (current_time - row['last_seen']) > timeout_threshold else 'Connected',
            axis=1
        )
