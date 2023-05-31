from typing import Any, Union, List

import numpy as np
import torch.nn
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader

from multi_modal_edge_ai.models.anomaly_detection.torch_models.torch_autoencoder import TorchAutoencoder
from multi_modal_edge_ai.commons.model import Model


class Autoencoder(Model):

    def __init__(self, encoder_dimensions: list[int], decoder_dimensions: list[int],
                 hidden_activation_fun: torch.nn.Module, output_activation_fun: torch.nn.Module) -> None:
        """
        This function will construct the autoencoder as specified in the function parameters and will set some fields
        regarding gpu/cpu and decision thresholds
        :param encoder_dimensions: A list with the size and dimensions of each layer of the encoder. Each entry in the
        list is a new layer, and its value the width in neurons of the layer
        :param decoder_dimensions: A list with the size and dimensions of each layer of the decoder part. This is most
        likely going to be the reverse of encoder_dimensions but not necessarily so.
        :param hidden_activation_fun: The activation function to be used for the hidden layers, most likely ReLu
        :param output_activation_fun: The activation function to be used for the hidden layers, most likely Sigmoid
        """
        self.loss_function = torch.nn.MSELoss()
        self.model = TorchAutoencoder(encoder_dimensions, decoder_dimensions, hidden_activation_fun,
                                      output_activation_fun)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.reconstruction_errors: list[float] = []
        self.reconstruction_loss_threshold = -1.0

    def train(self, data: Union[DataLoader[Any], List], **hyperparams: Any) -> list[float]:
        """
        This function will perform the entire training procedure on the autoencoder with the data provided
        :param data: The data on which to perform the training procedure
        :param hyperparams: The training hyperparameters: loss_function/learning_rate/epochs, etc...
        :return: A list with the average training losses of each epoch
        """
        assert isinstance(data, DataLoader), "Data must be of type DataLoader for the Autoencoder model"

        self.loss_function = hyperparams['loss_function']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=hyperparams["learning_rate"],
                                     weight_decay=1e-8)
        curr_reconstruction_errors = []
        avg_training_loss = []

        for epoch in range(hyperparams['n_epochs']):
            epoch_training_loss = []
            for windows in data:
                windows = windows.to(self.device).float()
                reconstructed_window = self.model(windows)

                loss = self.loss_function(reconstructed_window, windows)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_training_loss.append(loss)
            curr_reconstruction_errors += epoch_training_loss
            avg_training_loss.append(sum(epoch_training_loss) / len(epoch_training_loss))

        self.reconstruction_errors += torch.tensor(curr_reconstruction_errors, device='cpu').tolist()
        return avg_training_loss

    def set_reconstruction_error_threshold(self, quantile: float = 0.99) -> None:
        """
        This function will set the reconstruction error threshold, which is used on prediction time, according to the
        specified quantile on the errors seen in training
        :param quantile: The quantile of the threshold, e.g., 0.99
        """
        quantile_value = np.quantile(self.reconstruction_errors, quantile)
        self.reconstruction_loss_threshold = float(quantile_value)

    def predict(self, instance: Union[Tensor, DataFrame]) -> int:
        """
        This function will perform a forward pass on the instance provided. If the reconstruction error of the
        autoencoder is superior to the threshold set, it will return 0 for anomaly, and 1 otherwise. If the threshold
        has not been set, it will throw a NotImplementedError
        :param instance: The instance on which to perform the forward pass
        :return: 0 for anomaly, 1 for normal
        """
        self.model.eval()  # turn the model into evaluation mode
        instance = instance.to(self.device)  # send device to gpu if needed

        assert self.reconstruction_loss_threshold == -1, "You must set the reconstruction error before using prediction"

        with torch.no_grad():  # no need to construct the computation graph
            reconstructed = self.model(instance)
            return int(self.loss_function(reconstructed, instance) <= self.reconstruction_loss_threshold)

    def save(self, file_path: str) -> None:
        """
        This function will save the torch autoencoder on the specified path
        :param file_path: the file path
        """
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path: str) -> None:
        """
        This function will load the torch autoencoder from the specified path
        :param file_path: the file path
        """
        self.model.load_state_dict(torch.load(file_path))
