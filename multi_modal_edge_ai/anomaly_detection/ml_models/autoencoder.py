from typing import Any, Union, List

import torch.nn
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader

from multi_modal_edge_ai.anomaly_detection.torch_models.pytorch_autoencoder import PytorchAutoencoder
from multi_modal_edge_ai.commons.model import Model


class Autoencoder(Model):

    def __init__(self, encoder_dimensions: list[int], decoder_dimensions: list[int],
                 hidden_layers_activation_function: torch.nn.Module, output_layer_activation_function: torch.nn.Module,
                 loss_limit: float) -> None:
        self.model = PytorchAutoencoder(encoder_dimensions, decoder_dimensions, hidden_layers_activation_function,
                                        output_layer_activation_function)
        self.loss_function = torch.nn.MSELoss()
        self.loss_limit = loss_limit

    def train(self, dataset: Union[DataLoader[Any], List], **hyperparams: Any) -> list[float]:
        self.loss_function = hyperparams.get('loss_function', self.loss_function)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=hyperparams.get('learning_rate', 0.1),
                                     weight_decay=1e-8)

        avg_training_loss = []

        for epoch in range(hyperparams.get('epochs', 10)):
            epoch_training_loss = []
            for (window, _) in dataset:
                reconstructed_window = self.model(window)

                loss = self.loss_function(reconstructed_window, window)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_training_loss.append(loss)
            avg_training_loss.append(sum(epoch_training_loss) / len(epoch_training_loss))
        return avg_training_loss

    def predict(self, instance: Union[Tensor, DataFrame]) -> int:
        reconstructed = self.model(instance)
        return self.loss_function(reconstructed, instance) > 1

    def save(self, file_path: str) -> None:
        pass

    def load(self, file_path: str) -> None:
        pass

    @property
    def loss_limit(self):
        return self.loss_function

    @loss_limit.setter
    def loss_limit(self, value: float):
        self.loss_function = value

# class MyDataset(Dataset):
#     def __init__(self, df):
#         self.df = df
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         # Assuming that there are two columns 'features' and 'labels'
#         features = self.df.iloc[idx]
#         return features, features
#
#
# if __name__ == "__main__":
#     def clean(window):
#         return torch.Tensor([1] * 30)
#
#
#     adl_df = parse_file_with_idle(
#         "/home/rafael/TUDelft/cse/year2/q4/software-project/multi-modal-edge-ai/multi_modal_edge_ai/public_datasets/Aruba_Idle_Squashed.csv")
#     adl_df = split_into_windows(adl_df, 10, 3)
#     ae = Autoencoder([30, 18, 12, 8], [8, 12, 18, 30], torch.nn.ReLU(), torch.nn.Sigmoid())
#     df = adl_df.apply(clean, axis=1)
#     dataloader = DataLoader(MyDataset(df), batch_size=32, shuffle=True)
#     ae.train(dataloader, learning_rate=0.01)
#     print("trained")
