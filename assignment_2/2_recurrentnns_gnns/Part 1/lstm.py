"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import math


# python train.py --seed 1 --model_type LSTM --device cpu --train_steps 2000 --learning_rate 0.001 --batch_size 128 --num_hidden 128 --input_length 16



class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTM, self).__init__()
        #super().__init__()

        input_dim = int(0.25*hidden_dim) # as suggested by Philiph on Piazza, setting the embedding size to 1/4 of the hidden_dim

        self.embeddings = nn.Embedding(seq_length, input_dim)
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.device = device

        # Input modulation gate
        self.W_gx = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_gh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_g = nn.Parameter(torch.Tensor(hidden_dim))

        # Input gate
        self.W_ix = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_ih = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.Tensor(hidden_dim))

        # Forget gate
        self.W_fx = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_fh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(hidden_dim))

        # Output gate
        self.W_ox = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_oh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.Tensor(hidden_dim))

        # Output calculation
        self.W_ph = nn.Parameter(torch.Tensor(hidden_dim, num_classes))
        self.b_p = nn.Parameter(torch.Tensor(num_classes))

        self.kaiming_init()
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def kaiming_init(self):
        for p in self.parameters():
            if len(p.size()) == 2:
                p.data.normal_(0, 1 / math.sqrt(self.hidden_dim))
            else:
                p.data.fill_(0)

    def forward(self, x):

        h_t = torch.zeros(self.hidden_dim).to(self.device)
        c_t = torch.zeros(self.hidden_dim).to(self.device)
        x = self.embeddings(x.long())

        for time in range(self.seq_length - 1):
            x_t = x[:, time, :]
            # Input modulation gate
            g_t = torch.tanh(x_t @ self.W_gx + h_t @ self.W_gh + self.b_g)
            # Input gate
            i_t = torch.sigmoid(x_t @ self.W_ix + h_t @ self.W_ih + self.b_i)
            # Forget gate
            f_t = torch.sigmoid(x_t @ self.W_fx + h_t @ self.W_fh + self.b_f)
            # Output gate
            o_t = torch.sigmoid(x_t @ self.W_ox + h_t @ self.W_oh + self.b_o)
            # Cell state
            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t

        out = self.logSoftmax((h_t @ self.W_ph + self.b_p))
        return out
