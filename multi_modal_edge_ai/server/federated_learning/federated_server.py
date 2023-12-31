from typing import Any

import flwr as fl
import flwr.common
from flwr.common import Metrics
from flwr.server import ServerConfig

from multi_modal_edge_ai.server.federated_learning.PersistentFedAvg import PersistentFedAvg
from multi_modal_edge_ai.server.object_keepers.clients_keeper import ClientsKeeper
from multi_modal_edge_ai.server.object_keepers.models_keeper import ModelsKeeper


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """
    This function will perform the aggregation of the many metrics returned by each individual client.
    :param metrics: The list of metrics returned by each client
    :return: One single object of metrics with its values as the aggregation of all the clients' metrics
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    recalls = [num_examples * float(m["recall"]) for num_examples, m in metrics]
    precisions = [num_examples * float(m["precision"]) for num_examples, m in metrics]
    f1_scores = [num_examples * float(m["f1_score"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples), "recall": sum(recalls) / sum(examples),
            "precision": sum(precisions) / sum(examples), "f1_score": sum(f1_scores) / sum(examples)}


def training_loss_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """
    This will aggregate the training time metrics
    :param metrics: The list of metrics returned by each client
    :return: The aggregated object
    """
    training_loss_avg = [num_examples * float(m['avg_reconstruction_loss']) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"avg_reconstruction_loss": sum(training_loss_avg) / sum(examples)}


class FederatedServer:

    def __init__(self, server_address: str, models_keeper: ModelsKeeper, clients_keeper: ClientsKeeper) -> None:
        self.server_address = server_address
        self.models_keeper = models_keeper
        self.clients_keeper = clients_keeper

    def start_server(self, config: dict[str, Any], log_file_path: str, running_federation_workload: bool) -> None:
        """
        This function will start the rpc server with the specified parameters.
        :param config: The config which contains hyperparameters for both training, evaluation, and federation
        :param log_file_path: The path to the log file which Flower will use
        :param running_federation_workload: A boolean representing if it is a federation workload or a personalization
        workload
        :return:
        """
        strategy = PersistentFedAvg(fraction_fit=config["fraction_fit"], fraction_evaluate=config["fraction_evaluate"],
                                    min_fit_clients=config["min_fit_clients"],
                                    min_evaluate_clients=config["min_evaluate_clients"],
                                    min_available_clients=config["min_available_clients"],
                                    on_fit_config_fn=lambda _: config, on_evaluate_config_fn=lambda _: config,
                                    accept_failures=False, fit_metrics_aggregation_fn=training_loss_average,
                                    evaluate_metrics_aggregation_fn=weighted_average,
                                    running_federation_workload=running_federation_workload,
                                    clients_keeper=self.clients_keeper, models_keeper=self.models_keeper)
        flwr.common.configure("server", log_file_path)
        fl.server.start_server(
            server_address=self.server_address,
            strategy=strategy,
            config=ServerConfig(num_rounds=int(config["num_rounds"])),
        )

# This piece of code was used in order to test the interaction between federated client and server
# if __name__ == "__main__":
#     fs = FederatedServer("127.0.0.1:8080")
#     fs.start_server({"num_rounds": 2, "window_size": 10, "window_slide": 2, "one-hot": True, "batch_size": 32,
#                      "learning_rate": 0.01, "n_epochs": 2, "verbose": True, "event_based": True,
#                      "anomaly_whisker": 1.75, "clean_test_data_ratio": 0.2, "anomaly_generation_ratio": 0.1,
#                      "reconstruction_error_quantile": 0.99},
#                     "multi-modal-edge-ai/multi_modal_edge_ai/server/federated_learning/server_log")
