from collections import OrderedDict

import flwr as fl
import torch.nn

from multi_modal_edge_ai.client.federated_learning.train_and_eval import TrainEval
from multi_modal_edge_ai.commons.model import Model


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model: Model, train_eval: TrainEval):
        self.model = model
        self.train_eval = train_eval

    def get_parameters(self, config):
        if isinstance(self.model.model, torch.nn.Module):
            return [val.cpu().numpy() for _, val in self.model.model.state_dict().items()]
        else:
            params = self.model.model.get_params()
            return params

    def set_parameters(self, parameters):
        if isinstance(self.model.model, torch.nn.Module):
            params_dict = zip(self.model.model.state_dict().keys(), parameters())
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.model.load_state_dict(state_dict, strict=True)
        else:
            self.model.model.set_params(**parameters)

        self.model.save("multi_modal_edge_ai/client/anomaly_detection/anomaly_detection_model")

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        training_stats = self.train_eval.train(self.model, config)
        return self.get_parameters(config={}), *training_stats

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, df_size, evaluation_stats = self.train_eval.evaluate(self.model, config)
        return loss, df_size, evaluation_stats
