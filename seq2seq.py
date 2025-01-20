import torch
from torch import nn
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

X_in = torch.randint(0, 1234, (128, 20))
X_out = torch.randint(0, 1200, (128, 24))


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, X):
        X = self.embedding(X)
        _, (h, C) = self.lstm(X)
        return h, C


class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers):
        super().__init__()

        self.output_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, X, h, C):
        X = X.unsqueeze(1)
        X = self.embedding(X)
        out, (h, C) = self.lstm(X, (h, C))
        return self.fc_out(out.squeeze(1)), h, C


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        h, C = self.encoder(src)

        inp = trg[:, 0]

        for t in range(1, trg_len):
            out, h, C = self.decoder(inp, h, C)

            outputs[:, t - 1, :] = out

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = out.argmax(1)
            inp = trg[:, t] if teacher_force else top1

        return outputs


encoder = Encoder(1234, 1000, 1000, 4)

decoder = Decoder(1200, 1000, 1000, 4)

seq2seq = Seq2Seq(encoder, decoder, device)

logits = seq2seq(X_in, X_out)
