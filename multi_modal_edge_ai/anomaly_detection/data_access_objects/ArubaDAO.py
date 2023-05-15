import torch
from torch.utils.data import Dataset
import pandas as pd

class ArubaDAO(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.data.iloc[index]
        return sample