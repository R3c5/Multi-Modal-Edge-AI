import torch.nn


class PytorchAutoencoder(torch.nn.Module):
    def __init__(self, encoder_dimensions: list[int], decoder_dimensions: list[int],
                 hidden_layers_activation_function: torch.nn.Module,
                 output_layer_activation_function: torch.nn.Module) -> None:
        super().__init__()

        encoder_layers = []
        for i in range(1, len(encoder_dimensions)):
            encoder_layers.append(torch.nn.Linear(encoder_dimensions[i - 1], encoder_dimensions[i]))
            if i != (len(encoder_dimensions) - 1):
                encoder_layers.append(hidden_layers_activation_function)

        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(1, len(decoder_dimensions)):
            decoder_layers.append(torch.nn.Linear(decoder_dimensions[i - 1], decoder_dimensions[i]))
            decoder_layers.append(hidden_layers_activation_function if i != (
                len(decoder_dimensions) - 1) else output_layer_activation_function)

        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
