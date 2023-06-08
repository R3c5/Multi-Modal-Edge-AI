from collections import OrderedDict

import flwr as fl
import torch.nn
from pymongo.collection import Collection

from multi_modal_edge_ai.commons.model import Model


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model: Model, database_collection: Collection, train_fun, evaluate_fun):
        self.model = model
        self.database_collection = database_collection
        self.train_fun = train_fun
        self.evaluate_fun = evaluate_fun

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
        training_stats = self.train_fun(self.model, self.database_collection, config)
        return self.get_parameters(config={}), *training_stats

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, evaluation_stats = self.evaluate_fun(self.model, self.database_collection, config)
        return loss, *evaluation_stats
