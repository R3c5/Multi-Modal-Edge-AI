from typing import Union, Any, List

import numpy as np
import torch.nn as nn
import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader

from multi_modal_edge_ai.adl_inference.preprocessing.cnn_preprocess import cnn_format_dataset, cnn_format_input
from multi_modal_edge_ai.adl_inference.preprocessing.encoder import Encoder
from multi_modal_edge_ai.adl_inference.torch_models.pytorch_cnn import PytorchCNN
from multi_modal_edge_ai.commons.model import Model


class CNNModel(Model):
    def __init__(self, sensor_names: list[str], input_size: int, num_conv_layers: int, fc_dim: list[int],
                 hidden_activation_fun: nn.Module, output_activation_fun: nn.Module) -> None:
        """
        This function initializes the CNN network. It will create all the layers with their activation functions
        :param sensor_names: list of strings representing the names of the sensors used
        :param input_size: integer representing the size of the input to the CNN
        :param num_conv_layers: number of convolution layers
        :param fc_dim: A list with the size and dimensions of each layer of the fully connected part.
        The first layer created will take in_features the output size of the convolution layers
        and out_features the first element in the fc_dim. Note here that the last entry in fc_dimensions
        should be the number of classes classified
        :param hidden_activation_fun: The activation function to be used for the hidden
        layers, most likely ReLu
        :param output_activation_fun: The activation function to be used for the hidden
        layers, most likely Softmax
        """
        self.model = PytorchCNN(input_size, num_conv_layers, fc_dim, hidden_activation_fun, output_activation_fun)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.loss_function = torch.nn.MSELoss()
        self.num_sensors = len(sensor_names)
        self.sensor_encoder = Encoder(sensor_names)

    def train(self, dataset: Union[DataLoader[Any], List], **hyperparams: Any) -> None:
        """
        This function will perform the entire training procedure on the autoencoder with the data provided
        :param dataset: A list of windows as described in window_splitter.py
        :param hyperparams: The training hyperparameters: loss_function/learning_rate/epochs, etc...
        """
        data = cnn_format_dataset(dataset, self.num_sensors, self.sensor_encoder)

        self.loss_function = hyperparams.get('loss_function', self.loss_function)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=hyperparams.get('learning_rate', 0.01),
                                     weight_decay=1e-8)

        for epoch in range(hyperparams.get('epochs', 10)):
            for (input, label) in data:
                output = self.model(input)
                loss = self.loss_function(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, instance: Union[Tensor, DataFrame], window_start=None, window_end=None) -> Any:
        """
        This function will perform a forward pass on the instance provided. If the reconstruction error of the
        autoencoder is superior to the threshold set, it will return 0 for anomaly, and 1 otherwise. If the threshold
        has not been set, it will throw a NotImplementedError
        :param instance: The instance on which to perform the forward pass
        :param window_start: datetime representing the start time of the window,
        if None, the earliest sensor start time will be taken
        :param window_end: datetime representing the end time of the window,
        if None, the latest sensor end time will be taken
        :return: the encoded label of the predicted activity
        """
        # Reformat the instance so it can work on the CNN
        if window_start is None:
            window_start = np.min(instance['Start_Time'])

        if window_end is None:
            window_end = np.max(instance['End_Time'])

        instance = cnn_format_input(instance, window_start, window_end, self.num_sensors, self.sensor_encoder)

        # turn the model into evaluation mode
        self.model.eval()

        # Transform instance into a Pytorch tensor with type unsigned int 8, since values in the 2d array are 0s and 1s
        instance = torch.tensor(instance, dtype=torch.uint8).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(instance)

        # Return the prediction (you may need to process the output depending on your use case)
        return output.item()

    def save(self, file_path: str) -> None:
        """
        This function will save the torch CNN on the specified path
        :param file_path: the file path
        """
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path: str) -> None:
        """
        This function will load the torch CNN from the specified path
        :param file_path: the file path
        """
        self.model.load_state_dict(torch.load(file_path))
