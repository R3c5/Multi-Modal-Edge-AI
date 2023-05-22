from typing import Union

import torch.nn as nn
from pandas import DataFrame
from torch import Tensor


class PytorchCNN(nn.Module):

    def __init__(self, input_size: int, num_conv_layers: int, fc_dim: list[int],
                 hidden_activation_fun: nn.Module, output_activation_fun: nn.Module) -> None:
        """
        This function initializes the CNN network. It will create all the layers with their activation functions

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
        super().__init__()

        # Create convolutional layers
        conv_layers: list[nn.Module] = []
        in_channels = input_size
        for i in range(num_conv_layers):
            # Calculate number of out_channels based on the in_channels
            out_channels = in_channels * 2
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            conv_layers.append(hidden_activation_fun)
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate the output size after convolutional layers
        conv_output_size = self.calculate_conv_output_size(input_size, num_conv_layers)

        # Create fully connected layers
        fc_layers: list[nn.Module] = []
        in_features = conv_output_size
        for i in range(len(fc_dim)):
            out_features = fc_dim[i]
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(hidden_activation_fun if i != (len(fc_dim) - 1) else output_activation_fun)
            in_features = out_features

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x: Union[Tensor, DataFrame]) -> Union[Tensor, DataFrame]:
        """
        This function performs the forward pass on the given instance, by first encoding and then returning the decoding
        of such encoding
        :param x: The instance to perform the forward pass on
        :return: The decoded encoding of the original instance
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    @staticmethod
    def calculate_conv_dimensions(input_size: int, num_conv_layers: int) -> list[int]:
        conv_dim = [input_size]
        for _ in range(num_conv_layers):
            conv_dim.append(conv_dim[-1] * 2)
        return conv_dim[1:]

    @staticmethod
    def calculate_conv_output_size(input_size: int, num_conv_layers: int) -> int:
        conv_output_size = input_size
        for _ in range(num_conv_layers):
            conv_output_size = (conv_output_size - 2) / 2 + 1
        return conv_output_size
