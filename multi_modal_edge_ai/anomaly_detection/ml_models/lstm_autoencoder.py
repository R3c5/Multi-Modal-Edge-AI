from typing import Any, Union, List

import numpy as np
import torch.nn
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader

from multi_modal_edge_ai.anomaly_detection.torch_models.torch_lstm_autoencoder import TorchLSTMAutoencoder
from multi_modal_edge_ai.commons.model import Model


class LSTMAutoencoder(Model):

    def __init__(self, n_features: int, seq_len: int, hidden_dim: int, encoder_layers: int, decoder_layers: int,
                 dropout_rate: float = 0.05) -> None:
        """
        Initialize the LSTM Autoencoder model.
        The model is moved to the GPU if available. Mean Squared Error (MSE) is used as the loss function.
        The reconstruction loss threshold is initially set to -1.0, and the list of reconstruction errors is initialized
        as an empty list.
        :param n_features: The number of input features per ADL
        :param seq_len: The length of the input sequence in ADLs (most likely window size)
        :param hidden_dim: The number of hidden units in the LSTM layers.
        :param encoder_layers: The number of layers in the encoder part of the autoencoder.
        :param decoder_layers: The number of layers in the decoder part of the autoencoder.
        :param dropout_rate: The dropout rate to be used in the LSTM layers. Default is 0.05.
        """
        self.model = TorchLSTMAutoencoder(n_features, seq_len, hidden_dim, encoder_layers, decoder_layers, dropout_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.n_features = n_features
        self.seq_len = seq_len
        self.loss_function = torch.nn.MSELoss()
        self.reconstruction_errors: list[float] = []
        self.reconstruction_loss_threshold = -1.0

    def train(self, data: Union[DataLoader[Any], List], **hyperparams: Any) -> list[float]:
        """
        This function will perform the entire training procedure on the lstm autoencoder with the data provided.
        :param data: The data on which to perform the training procedure. It is important to note that the data received
        should be of the size (number_of_instances, window_size * n_features_per_adl)
        :param hyperparams: The training hyperparameters: loss_function/learning_rate/epochs, etc...
        :return: A list with the average training losses of each epoch
        """
        self.loss_function = hyperparams.get('loss_function', self.loss_function)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=hyperparams.get('learning_rate', 0.1),
                                     weight_decay=1e-8)
        curr_reconstruction_errors = []
        avg_training_loss = []

        for epoch in range(hyperparams.get('epochs', 10)):
            epoch_training_loss = []
            for windows in data:
                windows = windows.reshape((1, self.seq_len, self.n_features)).to(self.device).float()
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
        lstm autoencoder is superior to the threshold set, it will return 0 for anomaly, and 1 otherwise. If the
        threshold has not been set, it will throw a NotImplementedError
        :param instance: The instance on which to perform the forward pass
        :return: 0 for anomaly, 1 for normal
        """
        self.model.eval()  # turn the model into evaluation mode
        instance = instance.to(self.device)  # send device to gpu if needed

        if self.reconstruction_loss_threshold == -1:
            raise NotImplementedError("You must set the reconstruction error before using prediction")

        with torch.no_grad():  # no need to construct the computation graph
            reconstructed = self.model(instance)
            return int(self.loss_function(reconstructed, instance) <= self.reconstruction_loss_threshold)

    def save(self, file_path: str) -> None:
        """
        This function will save the torch lstm autoencoder on the specified path
        :param file_path: the file path
        """
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path: str) -> None:
        """
        This function will load the torch lstm autoencoder from the specified path
        :param file_path: the file path
        """
        self.model.load_state_dict(torch.load(file_path))
