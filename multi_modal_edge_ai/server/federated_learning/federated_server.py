import flwr as fl
import flwr.common
from flwr.common import Metrics
from flwr.common import Scalar
from flwr.server import ServerConfig

from multi_modal_edge_ai.server.federated_learning.SaveModelFedAvg import SaveModelFedAvg
from multi_modal_edge_ai.server.object_keepers.clients_keeper import ClientsKeeper
from multi_modal_edge_ai.server.object_keepers.models_keeper import ModelsKeeper


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    recalls = [num_examples * float(m["recall"]) for num_examples, m in metrics]
    precisions = [num_examples * float(m["precision"]) for num_examples, m in metrics]
    f1_scores = [num_examples * float(m["f1_score"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples), "recall": sum(recalls) / sum(examples),
            "precision": sum(precisions) / sum(examples), "f1_score": sum(f1_scores) / sum(examples)}


class FederatedServer:

    def __init__(self, server_address: str, models_keeper: ModelsKeeper, clients_keeper: ClientsKeeper) -> None:
        self.server_address = server_address
        self.models_keeper = models_keeper
        self.clients_keeper = clients_keeper

    def start_server(self, config: dict[str, Scalar], log_file_path):
        """
        This function will start the rpc server with the specified parameters.
        :param config: The config which contains hyperparameters for both training and evaluation
        :param log_file_path: The path to the log file which Flower will use
        :return:
        """
        strategy = SaveModelFedAvg(on_fit_config_fn=lambda _: config, on_evaluate_config_fn=lambda _: config,
                                   accept_failures=False, evaluate_metrics_aggregation_fn=weighted_average,
                                   clients_keeper=self.clients_keeper, models_keeper=self.models_keeper)
        flwr.common.configure("server", log_file_path)
        fl.server.start_server(
            server_address=self.server_address,
            strategy=strategy,
            config=ServerConfig(num_rounds=int(config["num_rounds"])),
        )

# if __name__ == "__main__":
#     fs = FederatedServer("127.0.0.1:8080")
#     fs.start_server({"num_rounds": 2, "window_size": 10, "window_slide": 2, "one-hot": True, "batch_size": 32,
#                      "learning_rate": 0.01, "n_epochs": 2, "verbose": True, "event_based": True,
#                      "anomaly_whisker": 1.75, "clean_test_data_ratio": 0.2, "anomaly_generation_ratio": 0.1,
#                      "reconstruction_error_quantile": 0.99},
#                     "multi-modal-edge-ai/multi_modal_edge_ai/server/federated_learning/server_log")
