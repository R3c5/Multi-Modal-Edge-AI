from abc import ABC, abstractmethod
from typing import Any, Union, List

from torch.utils.data import DataLoader
from torch import Tensor
from pandas import DataFrame


class Model(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self, data: Union[DataLoader[Any], List], **hyperparams: Any) -> Any:
        """
        abstract method in order to train a model on a dataset, with any hyperparams needed
        """
        pass

    @abstractmethod
    def predict(self, instance: Union[Tensor, DataFrame]) -> Any:
        """
        abstract method to predict an instance using the model
        """
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        """
        abstract method to save the model at a given file path
        """
        pass

    @abstractmethod
    def load(self, file_path: str) -> None:
        """
        abstract method to load the model from a given file path
        """
        pass
