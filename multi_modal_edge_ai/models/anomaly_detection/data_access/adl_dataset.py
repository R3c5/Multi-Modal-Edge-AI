import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Any, Callable, Optional


class ADLDataset(Dataset[torch.Tensor]):
    def __init__(self, data: pd.DataFrame, transform: Optional[Callable[..., Any]] = None) -> None:
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.data.iloc[index]
        return torch.Tensor(sample)
