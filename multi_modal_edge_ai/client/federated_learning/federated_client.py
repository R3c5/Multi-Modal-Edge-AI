import flwr as fl

from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper
from multi_modal_edge_ai.client.federated_learning.flower_clients import FlowerClient
from multi_modal_edge_ai.client.federated_learning.train_and_eval import TrainEval
from multi_modal_edge_ai.commons.model import Model


class FederatedClient:

    def __init__(self, model_keeper: ModelKeeper, train_eval: TrainEval) -> None:
        """
        Constructor for the basic Federated Client
        :param model_keeper: The model keeper which holds the machine learning model
        :param train_eval: The train eval object with the training, evaluation and other data
        """
        self.model_keeper = model_keeper
        self.flower_client = FlowerClient(model_keeper, train_eval)

    def start_numpy_client(self, server_address):
        """
        This function will start the client. It will be open to connection from the server
        :param server_address: The address of the Flower server
        :return:
        """
        fl.client.start_numpy_client(
            server_address=server_address,
            client=self.flower_client
        )
