"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_classes, act_fn = nn.ELU(), use_dropout = False, use_batch_norm = False):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """
        super().__init__()

        layers = []
        layer_sizes = [n_inputs] + n_hidden  # + [n_hidden]

        for layer_id in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[layer_id - 1], layer_sizes[layer_id]),
                       nn.ELU()]
            if use_batch_norm == True:
                layers += [nn.BatchNorm1d(layer_sizes[layer_id])]

        if use_dropout==True:
            layers += [nn.Dropout(p = 0.1)]

        layers += [nn.Linear(layer_sizes[-1], n_classes)]
        self.layers = nn.ModuleList(
            layers)  # A module list registers a list of modules as submodules (e.g. for parameters)

    def forward(self, x):
        out = torch.tensor(x)
        for layer in self.layers:
            out = layer(out)
        return out



