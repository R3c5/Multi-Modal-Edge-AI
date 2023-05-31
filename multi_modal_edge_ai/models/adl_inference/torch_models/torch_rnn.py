from typing import Union

import torch
import torch.nn as nn
from torch import Tensor


class TorchRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int, non_linearity: str, last_layer_activation: Union[torch.nn.Module, None]) -> None:
        """
        Initialize the TorchRNN class
        :param input_size: the size of each input vector
        :param hidden_size: the size of the hidden state
        :param num_layers: the number of hidden layers in the RNN
        :param num_classes: the number of classes to classify
        :param non_linearity: The non-linearity to use. Can be either 'tanh' or 'relu'.
        :param last_layer_activation: The activation function to use in the last layer.
        """
        super(TorchRNN, self).__init__()
        self.non_linearity = non_linearity
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity=non_linearity, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.last_layer_activation = last_layer_activation

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the RNN
        :param x: the input vector
        :return: the output vector
        """
        # Set initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        if self.last_layer_activation is not None:
            out = self.last_layer_activation(out)

        return out
