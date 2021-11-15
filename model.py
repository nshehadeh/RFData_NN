import torch
import os
from torch import nn


"""
- class extends nn.Module
- creates layers, activation function, 
    drop out & batch normalization, initializes weights
- defines forward function
"""


class FullConnectedNet(nn.Module):
    def __init__(self, input_dim, output_dim, layer_width, num_hidden=1,
                 dropout=0, dropout_input=0, starting_weights=None,
                 batch_norm_enable=False):
        super().__init__()  # has to do with inheritance

        # first layer, inputs
        self.layers = nn.ModuleList([nn.Linear(input_dim, layer_width)])
        # connect hidden layers
        for i in range(num_hidden - 1):
            self.layers.append(nn.Linear(layer_width, layer_width))
        # connect last layer, output
        self.layers.append(nn.Linear(layer_width, output_dim))

        # create activation function
        self.relu = nn.ReLU()

        # dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout_input = nn.Dropout(dropout_input)

        # batch normalization ifi batch_norm_enable is TRUE
        self.batch_norm_enable = batch_norm_enable
        if batch_norm_enable:
            self.bn_layers = nn.ModuleList([])
        for i in range(num_hidden):
            self.bn_layers.append(nn.BatchNorm1d(layer_width))

        # initialize weights randomly
        for i in range(len(self.layers)):
            nn.init.kaiming_normal_(self.layers[i].weight.data)
            self.layers[i].bias.data.fill_(0.01)

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            if self.batch_norm_enable:
                x = self.bn_layers[i](x)
            x = self.relu(x)
            x = self.dropout(x)
        # last layer
        x = self.layers[-1](x)
        return x

