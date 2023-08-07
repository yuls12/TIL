import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Decoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Decoder, self).__init__()

        # Be aware of value of 'batch_first' parameter and 'bidirectional' parameter.
        self.rnn = nn.LSTM(
            word_vec_size + hidden_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=False,
            batch_first=True,
        )

    def forward(self, emb_t, h_t_1_tilde, h_t_1):  # 한 time-step씩 들어오게 된다 가정.
        # |emb_t| = (batch_size, 1, word_vec_size)
        # |h_t_1_tilde| = (batch_size, 1, hidden_size)
        # |h_t_1[0]| = (n_layers, batch_size, hidden_size)
        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)

        if h_t_1_tilde is None:
            # If this is the first time-step,
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_()  # 초기화

        # Input feeding trick.
        x = torch.cat([emb_t, h_t_1_tilde], dim=-1)
        # |x| = (batch_size, 1, word_vec_size + hidden_size)

        # Unlike encoder, decoder must take an input for sequentially.
        y, h = self.rnn(x, h_t_1)
        # |y| = (batch_size, 1, hidden_size)
        # |h| = (n_layers, batch_size, hidden_size)
        return y, h
