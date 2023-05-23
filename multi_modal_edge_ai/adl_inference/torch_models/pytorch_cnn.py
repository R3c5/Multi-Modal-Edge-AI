from typing import Union

import torch.nn as nn
from pandas import DataFrame
from torch import Tensor


class PytorchCNN(nn.Module):
    def __init__(self, num_conv_layers: int, num_fc_layers: int,
                 window_length: int, num_sensors: int, num_classes: int,
                 hidden_activation: nn.Module, output_activation: nn.Module):
        """
        This function initializes the CNN network. It will create all the layers with their activation functions
        :param num_conv_layers: integer representing the number of convolution layers
        :param num_fc_layers: integer representing the number of fully connected layers
        :param window_length: integer representing the size of the input
        :param num_sensors: integer representing the number of sensors expected
        :param num_classes: integer representing the number of classes that can be classified
        :param hidden_activation: The activation function to be used for the hidden
        layers, most likely ReLu
        :param output_activation: The activation function to be used for the hidden
        layers, most likely Softmax
        """
        super(PytorchCNN, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers
        self.num_classes = num_classes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # Convolution layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1))
        self.conv_layers.append(hidden_activation)
        self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for _ in range(self.num_conv_layers - 1):
            self.conv_layers.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
            self.conv_layers.append(hidden_activation)
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(window_length // 4, 128))
        self.fc_layers.append(hidden_activation)
        for _ in range(self.num_fc_layers - 2):
            self.fc_layers.append(nn.Linear(128, 128))
            self.fc_layers.append(hidden_activation)

        # Output layer
        self.output_layer = nn.Linear(128, num_classes)
        self.output_activation = output_activation

    def forward(self, x: Tensor):
        """
        This function performs the forward pass on the given instance, by first encoding and then returning the decoding
        of such encoding
        :param x: The instance to perform the forward pass on
        :return: The decoded encoding of the original instance
        """
        x = self.apply_conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.apply_fc_layers(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x

    def apply_conv_layers(self, x: Tensor):
        """
        Calculate x after it goes through the convolution layers
        :param x: Tensor for the input before each conv layer
        :return: result after a convolution layer is applied
        """
        for layer in self.conv_layers:
            x = layer(x)
        return x

    def apply_fc_layers(self, x: Tensor):
        """
        Calculate x after it goes through the fully connected layers
        :param x: Tensor for the input before each fc layer
        :return: result after a convolution layer is applied
        """
        for layer in self.fc_layers:
            x = layer(x)
        return x
