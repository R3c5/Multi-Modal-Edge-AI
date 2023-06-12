import flwr as fl
from flwr.common import Metrics


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples), "recall": sum(recalls) / sum(examples),
            "precision": sum(precisions) / sum(examples), "f1_score": sum(f1_scores) / sum(examples)}


class FederatedServer:

    def __init__(self, server_address: str) -> None:
        self.server_address = server_address

    def start_server(self, config):
        fl.server.start_server(
            # server_address=self.server_address,
            config=fl.server.ServerConfig(num_rounds=1),
            strategy=fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
        )


if __name__ == "__main__":
    fs = FederatedServer("")
    fs.start_server({})
