from abc import ABC, abstractmethod
from typing import Any, Union
from torch.utils.data import DataLoader
from torch import Tensor
from pandas import DataFrame, Series


class Model(ABC):
    @abstractmethod
    def train(self, dataset: Union['DataLoader[Any]', DataFrame], **hyperparams: Any) -> None:
        pass

    @abstractmethod
    def predict(self, instance: Union[Tensor, 'Series[Any]']) -> Any:
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        pass
