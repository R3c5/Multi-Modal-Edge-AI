import flwr as fl

from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper
from multi_modal_edge_ai.client.federated_learning.flower_clients import FlowerClient
from multi_modal_edge_ai.client.federated_learning.train_and_eval import TrainEval


class FederatedClient:

    def __init__(self, model_keeper: ModelKeeper, train_eval: TrainEval, federation_workload: bool) -> None:
        """
        Constructor for the basic Federated Client
        :param model_keeper: The model keeper which holds the machine learning model
        :param train_eval: The train eval object with the training, evaluation and other data
        :param federation_workload: The boolean representing whether it is a federation or a personalization workload
        """
        self.model_keeper = model_keeper
        self.flower_client = FlowerClient(model_keeper, train_eval, federation_workload)

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

# This piece of code was used in order to test the interaction between federated client and server
# if __name__ == "__main__":
#     adl_df = parse_file_with_idle(
#         "multi-modal-edge-ai/tests/models/anomaly_detection/dummy_datasets/dummy_aruba.csv")
#     distinct_adl_list = pd.unique(adl_df.iloc[:, 2::3].values.ravel('K'))
#     model_keepr = ModelKeeper(Autoencoder([120, 80], [80, 120], torch.nn.ReLU(), torch.nn.Sigmoid()),
#                               "multi-modal-edge-ai/multi_modal_edge_ai/client/anomaly_detection/anomaly_detection_model"
#                               )
#     collection = get_collection(get_database(get_database_client(), "coho-edge-ai"), "adl_test")
#     train_eva = TrainEval(collection, distinct_adl_list, MinMaxScaler().fit([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))
#     fc = FederatedClient(model_keepr, train_eva)
#     fc.start_numpy_client("127.0.0.1:8080")
