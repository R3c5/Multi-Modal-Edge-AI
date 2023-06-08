import flwr as fl

class SaveModelStrategy(fl.server.strategy.FedAvg)

def start_federated_server(config: fl.server.ServerConfig) -> ...:
    strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

    fl.server.start_server(config=config)
