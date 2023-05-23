from torch import nn


class TorchLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.05):
        super(TorchLSTMAutoencoder, self).__init__()

        # Define Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim,
                               batch_first=True, dropout=dropout_rate)

        # Define Decoder
        self.decoder = nn.LSTM(hidden_dim, input_dim,
                               batch_first=True, dropout=dropout_rate)

    def forward(self, input):
        # Encoding
        outputs, (hidden, cell) = self.encoder(input)

        # Decoding
        decoder_outputs, (_, _) = self.decoder(hidden)
        return decoder_outputs
