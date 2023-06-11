import flwr as fl



class FederatedServer:

    def __init__(self, server_address: str) -> None:
        self.server_address = server_address

    def start_server(self, config):
        fl.server.start_server(
            server_address=self.server_address,
        )