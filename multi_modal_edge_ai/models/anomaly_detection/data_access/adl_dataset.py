import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Any, Callable, Optional


class ADLDataset(Dataset[torch.Tensor]):
    """
    This class is destined to hold any sort of ADL sequence data. It isn't respective to any specific ADL sequence
    dataset to provide flexibility. It extends from Torch's Dataset so as to easily wrap it around a DataLoader for
    deep learning purposes.
    """
    def __init__(self, data: pd.DataFrame, transform: Optional[Callable[..., Any]] = None) -> None:
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.data.iloc[index]
        return torch.Tensor(sample)
