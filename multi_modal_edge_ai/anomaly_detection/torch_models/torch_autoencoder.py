import torch.nn
from torch import Tensor


class TorchAutoencoder(torch.nn.Module):
    def __init__(self, encoder_dim: list[int], decoder_dim: list[int], hidden_activation_fun: torch.nn.Module,
                 output_activation_fun: torch.nn.Module) -> None:
        """
        This function initializes the autoencoder network. It will create all the layers with their activation functions
        :param encoder_dim: A list with the size and dimensions of each layer of the encoder. Each entry in the
        list is a new layer, and its value the width in neurons of the layer
        :param decoder_dim: A list with the size and dimensions of each layer of the decoder part. This is most
        likely going to be the reverse of encoder_dimensions but not necessarily so.
        :param hidden_activation_fun: The activation function to be used for the hidden layers, most likely ReLu
        :param output_activation_fun: The activation function to be used for the hidden layers, most likely Sigmoid
        """
        super().__init__()

        encoder_layers: list[torch.nn.Module] = []
        for i in range(1, len(encoder_dim)):
            encoder_layers.append(torch.nn.Linear(encoder_dim[i - 1], encoder_dim[i]))
            if i != (len(encoder_dim) - 1):
                encoder_layers.append(hidden_activation_fun)

        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers: list[torch.nn.Module] = []
        for i in range(1, len(decoder_dim)):
            decoder_layers.append(torch.nn.Linear(decoder_dim[i - 1], decoder_dim[i]))
            decoder_layers.append(hidden_activation_fun if i != (
                    len(decoder_dim) - 1) else output_activation_fun)

        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        This function performs the forward pass on the given instance, by first encoding and then returning the decoding
        of such encoding
        :param x: The instance to perform the forward pass on
        :return: The decoded encoding of the original instance
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
