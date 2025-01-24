import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout, layer_norm_flag):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTM(
            input_dim, hidden_dim1, num_layers=2, batch_first=True, dropout=dropout
        )
        self.lstm2 = nn.LSTM(
            hidden_dim1, hidden_dim2, num_layers=2, batch_first=True, dropout=dropout
        )
        self.layer_norm_flag = layer_norm_flag
        self.layer_norm = nn.LayerNorm(hidden_dim2)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, (hidden, _) = self.lstm2(x)
        if self.layer_norm_flag:
            x = self.layer_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoded_dim, hidden_dim1, output_dim, dropout):
        super(Decoder, self).__init__()
        self.lstm1 = nn.LSTM(
            encoded_dim, hidden_dim1, num_layers=2, batch_first=True, dropout=dropout
        )
        self.lstm2 = nn.LSTM(
            hidden_dim1, output_dim, num_layers=2, batch_first=True, dropout=dropout
        )

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x


class LSTMAutoencoder(nn.Module):
    def __init__(
        self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout, layer_norm_flag
    ):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = Encoder(
            input_dim, hidden_dim1, hidden_dim2, dropout, layer_norm_flag
        )
        self.decoder = Decoder(hidden_dim2, hidden_dim1, output_dim, dropout)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, x):
        decoded = self.decoder(x)
        return decoded


class sm_Encoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=9):
        super(sm_Encoder, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm1(x)
        # y has the last hidden state repeated across the sequence length
        u = x[:, -1, :].unsqueeze(1).repeat(1, x.size(1), 1)
        return u


class sm_Decoder(nn.Module):
    def __init__(self, encoded_dim=9, hidden_dim=9, output_dim=1):
        super(sm_Decoder, self).__init__()
        self.lstm1 = nn.LSTM(encoded_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dense_out = nn.Linear(hidden_dim, output_dim)
        self.identity = nn.Identity()  # To show the correct output shape
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x):
        x, _ = self.lstm1(x)
        batch_size = x.shape[0]
        window = x.shape[1]
        x = x.reshape(-1, self.hidden_dim)
        x = self.dense_out(x)
        x = x.reshape(batch_size, window, self.output_dim)
        x = self.identity(x)
        return x


class sm_LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim1=9, hidden_dim2=9, output_dim=1):
        super(sm_LSTMAutoencoder, self).__init__()
        self.encoder = sm_Encoder(input_dim, hidden_dim1)
        self.decoder = sm_Decoder(hidden_dim1, hidden_dim2, output_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, x):
        decoded = self.decoder(x)
        return decoded
