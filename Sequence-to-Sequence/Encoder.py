import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Encoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Encoder, self).__init__()

        # Be aware of value of 'batch_first' parameter.
        # Also, its hidden_size is half of original hidden_size,
        # because it is bidirectional.
        self.rnn = nn.LSTM(
            word_vec_size,
            int(hidden_size / 2),
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, emb):
        # |emb| = (batch_size, length, word_vec_size)

        if isinstance(emb, tuple):
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first=True)  # padded sequence
        else:
            x = emb

        y, h = self.rnn(x)
        # |y| = (batch_size, length, hidden_size)
        # |h[0]| = (num_layers * 2, batch_size, hidden_size / 2)

        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)

        return y, h
