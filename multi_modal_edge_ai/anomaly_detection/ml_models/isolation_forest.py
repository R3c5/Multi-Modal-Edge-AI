import pickle
from typing import Any, Union, List

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.ensemble import IsolationForest
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from multi_modal_edge_ai.anomaly_detection.utils import dataloader_to_numpy
from multi_modal_edge_ai.commons.model import Model


class IForest(Model):

    def __init__(self) -> None:
        """
        Constructor for the IForest class. This method initializes a new instance of the IsolationForest model.

        The IForest model defaults to sklearn's IsolationForest with no specific parameters set.
        The parameters of the model can be customized during the training process.
        """
        self.model = IsolationForest()

    def train(self, data: Union[DataLoader[Any], List], **hyperparams: Any) -> None:
        """
        Train the IForest model on the provided data. This method uses the sklearn IsolationForest's fit method to train
        the model. The model's parameters can be customized by providing them in the `hyperparams` argument. The input
        data is transformed into a numpy array before training the model.
        :param data: the dataloader of the instances on which to train
        :param hyperparams: the hyperparameters to set the IForest to
        """
        assert isinstance(data, DataLoader), "Data must be of type DataLoader for the IForest model"

        i_forest_hparams = hyperparams["i_forest_hparams"]

        self.model.set_params(**i_forest_hparams)
        self.model.fit(dataloader_to_numpy(data))

    def predict(self, instance: Union[Tensor, DataFrame]) -> list[int]:
        """
        Perform anomaly detection on the provided instance using the trained IForest model.
        The method uses the sklearn IsolationForest's predict method to classify the instance.
        The instance is first converted to a numpy array, if it is not already.
        Anomalous instances are identified by a prediction of -1 from the IForest model,
        while normal instances are identified by a prediction of 1.
        The method returns a binary label where 0 indicates an anomaly and 1 indicates a normal instance.
        :param instance: the data instance on which to perform anomaly detection
        :return: a prediction for the provided data instance
        """
        # Ensure that instance is a 2D numpy array
        if isinstance(instance, DataFrame):
            instance = instance.values
        elif isinstance(instance, Tensor):
            instance = instance.cpu().numpy()

        prediction = self.model.predict(instance)
        return np.where(prediction == -1, 0, 1).tolist()[0]

    def save(self, file_path: str) -> None:
        """
        This function will save sklearn's IsolationForest on the specified path
        :param file_path: the file path
        """
        with open(file_path, "wb") as file:
            pickle.dump(self.model, file)

    def load(self, file_path: str) -> None:
        """
        This function will load sklearn's IsolationForest from the specified path
        :param file_path: the file path
        """
        with open(file_path, "rb") as file:
            self.model = pickle.load(file)
