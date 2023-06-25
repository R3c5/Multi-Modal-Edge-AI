from collections import OrderedDict
from typing import Any, Dict

import flwr as fl
import torch.nn
from flwr.common import Scalar

from multi_modal_edge_ai.client.federated_learning.train_and_eval import TrainEval


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model_keeper, train_eval: TrainEval, federation_workload: bool) -> None:
        """
        Constructor for the flower client
        :param model_keeper: The model keeper which holds the machine learning model
        :param train_eval: The train eval object with the training, evaluation and other data
        :param federation_workload: The boolean representing whether it is a federation or a personalization workload
        """
        self.model_keeper = model_keeper
        self.train_eval = train_eval
        self.federation_workload = federation_workload

    def get_parameters(self, config: dict[str, Scalar]) -> Any:
        """
        This function will get the parameters of the current model
        :param config: The config with which get the parameters
        :return: The parameters
        """
        if isinstance(self.model_keeper.model.model, torch.nn.Module):
            return [val.cpu().numpy() for _, val in self.model_keeper.model.model.state_dict().items()]
        else:
            params = self.model_keeper.model.model.get_params()
            return params

    def set_parameters(self, parameters) -> None:
        """
        This function will override the parameters of the current model
        :param parameters: The parameters to override
        :return:
        """
        if isinstance(self.model_keeper.model.model, torch.nn.Module):
            params_dict = zip(self.model_keeper.model.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model_keeper.model.model.load_state_dict(state_dict, strict=True)
        else:
            self.model_keeper.model.model.set_params(**parameters)

        self.model_keeper.save_model()

    def fit(self, parameters, config: dict[str, Scalar]) -> tuple[Any, int, dict[str, Scalar]]:
        """Any
        This function will fit the model on the local data
        :param parameters: The parameters on which to start training
        :param config: The config of the training procedure
        :return: The model parameters and the training stats
        """
        if self.federation_workload:
            self.set_parameters(parameters)
        training_stats = self.train_eval.train(self.model_keeper.model, config)
        return self.get_parameters(config={}), training_stats[0], training_stats[1]

    def evaluate(self, parameters, config: dict[str, Scalar]) -> tuple[float, int, Dict[str, Scalar]]:
        """
        This function will evaluate a model on the local data
        :param parameters: The parameters of the model to evaluate
        :param config: The config of the evaluation procedure
        :return: The loss, size of evaluation dataset and other stats
        """
        if self.federation_workload:
            self.set_parameters(parameters)
        loss, df_size, evaluation_stats = self.train_eval.evaluate(self.model_keeper.model, config)
        return loss, df_size, evaluation_stats
