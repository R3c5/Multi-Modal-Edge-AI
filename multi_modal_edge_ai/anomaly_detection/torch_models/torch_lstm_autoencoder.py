from typing import Union

from pandas import DataFrame
from torch import Tensor
from torch import nn


class TorchLSTMAutoencoder(nn.Module):

    def __init__(self, n_features: int, seq_len: int, hidden_dim: int, encoder_layers: int, decoder_layers: int,
                 dropout_rate: float = 0.05) -> None:
        """
        Initialize the LSTM Autoencoder model.
        An LSTM Autoencoder model is created with the provided parameters for the number of features, sequence length,
        hidden dimensions, and number of layers for both encoder and decoder. Dropout can also be specified.
        :param n_features: The number of input features per ADL
        :param seq_len: The length of the input sequence in ADLs (most likely window size)
        :param hidden_dim: The number of hidden units in the LSTM layers.
        :param encoder_layers: The number of layers in the encoder part of the autoencoder.
        :param decoder_layers: The number of layers in the decoder part of the autoencoder.
        :param dropout_rate: The dropout rate to be used in the LSTM layers. Default is 0.05.
        """
        super(TorchLSTMAutoencoder, self).__init__()

        self.n_features, self.seq_len = n_features, seq_len
        # Define Encoder
        self.encoder = nn.LSTM(n_features, hidden_dim, num_layers=encoder_layers,
                               batch_first=True, dropout=dropout_rate)

        # Define Decoder
        self.decoder = nn.LSTM(hidden_dim, n_features, num_layers=decoder_layers,
                               batch_first=True, dropout=dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        """
        This function performs the forward pass on the given instance, by first encoding and then returning the decoding
        of such encoding
        :param x: The instance to perform the forward pass on
        :return: The decoded encoding of the original instance
        """
        outputs, (_, _) = self.encoder(x)
        decoder_outputs, (_, _) = self.decoder(outputs)

        return decoder_outputs
