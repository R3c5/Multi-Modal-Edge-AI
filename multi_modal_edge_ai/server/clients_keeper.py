from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd


class ClientsKeeper:
    def __init__(self) -> None:
        self.client_columns = ['ip', 'status', 'last_seen', 'num_adls', 'num_anomalies']
        self.column_types: Dict[str, Any] = {'ip': str, 'status': str, 'last_seen': 'datetime64[s]', 'num_adls': int,
                                             'num_anomalies': int}
        df = pd.DataFrame(columns=self.client_columns)
        self.connected_clients = df.astype(self.column_types)

    def add_client(self, new_client: dict) -> None:
        """
        Add a new client to the connected_clients dataframe
        :param new_client: dictionary containing the ip, status, last_seen, num_adls and num_anomalies
        """
        self.connected_clients = pd.concat([self.connected_clients,
                                            pd.DataFrame([new_client], columns=self.client_columns)], ignore_index=True)

    def update_client(self, client: dict) -> bool:
        """
        Update the client in the connected_clients with the same ip as the one passed as a parameter
        :param client: dictionary containing the ip, status, last_seen, num_adls and num_anomalies
        :return True if update was successful and False if no client with the expected ip was found
        """
        ip = client['ip']
        if ip not in self.connected_clients['ip'].values:
            return False
        index = self.connected_clients.index[self.connected_clients['ip'] == ip].tolist()[0]

        # Update the values in the row
        self.connected_clients.loc[index, 'status'] = client['status']
        self.connected_clients.loc[index, 'last_seen'] = client['last_seen']
        self.connected_clients.loc[index, 'num_adls'] = client['num_adls']
        self.connected_clients.loc[index, 'num_anomalies'] = client['num_anomalies']
        return True

    def update_clients_statuses(self):
        current_time = datetime.now()
        timeout_threshold = timedelta(hours=3)

        self.connected_clients['status'] = self.connected_clients.apply(
            lambda row: 'Disconnected' if (current_time - row['last_seen']) > timeout_threshold else 'Connected',
            axis=1
        )
