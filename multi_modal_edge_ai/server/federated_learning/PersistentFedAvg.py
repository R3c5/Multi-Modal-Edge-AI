import pickle
from collections import OrderedDict
from datetime import datetime
from logging import INFO
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import torch
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    NDArrays,
    MetricsAggregationFn,
    parameters_to_ndarrays
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from multi_modal_edge_ai.server.object_keepers.clients_keeper import ClientsKeeper
from flwr.server.strategy.aggregate import aggregate


class PersistentFedAvg(FedAvg):

    def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            running_federation_workload: bool,
            clients_keeper: ClientsKeeper,
            models_keeper
    ):
        super().__init__(fraction_fit=fraction_fit,
                         fraction_evaluate=fraction_evaluate,
                         min_fit_clients=min_fit_clients,
                         min_evaluate_clients=min_evaluate_clients,
                         min_available_clients=min_available_clients,
                         evaluate_fn=evaluate_fn,
                         on_fit_config_fn=on_fit_config_fn,
                         on_evaluate_config_fn=on_evaluate_config_fn,
                         accept_failures=accept_failures,
                         initial_parameters=initial_parameters,
                         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                         evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)
        self.running_federation_workload = running_federation_workload
        self.clients_keeper = clients_keeper
        self.models_keeper = models_keeper

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        This function will perform the aggregation of the weights. It is very similar to the default from Flower's
        FedAvg, but here IPs of the client participants are extracted for dashboard update purposes
        :param server_round: The server round currently being aggregated
        :param results: The successful model results
        :param failures: The failures of the unsuccessful model results
        :return: The parameters and trainings statistics
        """

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        aggregated_weights_results = aggregate(weights_results)

        aggregated_parameters = super().aggregate_fit(server_round, results, failures)

        clients_ips = [client_proxy.cid for client_proxy, _ in results]
        aggregation_date = datetime.now()
        for ip in clients_ips:
            clean_ip = ip.split(':')[1]
            if self.running_federation_workload:
                self.clients_keeper.update_last_model_aggregation(clean_ip, aggregation_date)
            else:
                self.clients_keeper.update_last_model_personalization(clean_ip, aggregation_date)

        if aggregated_parameters is not None:
            if self.running_federation_workload:
                self.set_parameters(aggregated_weights_results)
                log(
                    INFO,
                    "aggregated %s clients successfully out of %s total selected clients",
                    len(results),
                    len(results) + len(failures)
                )
            else:
                log(
                    INFO,
                    "%s clients successfully personalized their local model",
                    len(results)
                )

        return aggregated_parameters

    def set_parameters(self, parameters) -> None:
        """
        This function will override the parameters of the current model
        :param parameters: The parameters to override
        :return:
        """
        if isinstance(self.models_keeper.anomaly_detection_model.model, torch.nn.Module):
            params_dict = zip(self.models_keeper.anomaly_detection_model.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.models_keeper.anomaly_detection_model.model.load_state_dict(state_dict, strict=True)
        else:
            self.models_keeper.anomaly_detection_model.model.set_params(**parameters)

        self.models_keeper.save_anomaly_detection_model()

    def get_parameters(self) -> Any:
        """
        This function will get the parameters of the current model
        :return: The parameters
        """
        if isinstance(self.models_keeper.anomaly_detection_model.model, torch.nn.Module):
            return [val.cpu().numpy() for _, val in
                    self.models_keeper.anomaly_detection_model.model.state_dict().items()]
        else:
            params = self.models_keeper.anomaly_detection_model.model.get_params()
            return params
