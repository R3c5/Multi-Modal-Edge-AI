import flwr as fl

from multi_modal_edge_ai.client.federated_learning.flower_clients import FlowerClient
from multi_modal_edge_ai.client.federated_learning.train_and_eval import TrainEval
from multi_modal_edge_ai.commons.model import Model


class FederatedClient:

    def __init__(self, model: Model, train_eval: TrainEval) -> None:
        self.model = model
        self.flower_client = FlowerClient(model, train_eval)

    def start_numpy_client(self, server_address):
        fl.client.start_numpy_client(
            server_address=server_address,
            client=self.flower_client
        )
