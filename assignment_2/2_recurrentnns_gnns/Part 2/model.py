# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0', embedding_dim=32, drop_prob=0.5):

        super(TextGenerationModel, self).__init__()

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.device = device

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_num_hidden, num_layers=lstm_num_layers,
                            dropout=drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(in_features=lstm_num_hidden, out_features=vocabulary_size)

        self.logSoftmax = nn.LogSoftmax(dim=2)
        self.embeddings = nn.Embedding(vocabulary_size, embedding_dim)

    def forward(self, x, h=None, temperature=1):
        x = self.embeddings(x)

        if h == None:
            lstm_output, h = self.lstm(x)  # randomly intialize the hidden state and cell state
        else:
            lstm_output, h = self.lstm(x, h)  # used for sampling from the model

        drop_output = self.dropout(lstm_output)
        fc_output = self.fc(drop_output)  # shape: [seq_length, batch_size, voc_size]

        return self.logSoftmax(
            fc_output * temperature), h  # shape: [seq_length, batch_size, voc_size], a disritbution over the vocabulary
