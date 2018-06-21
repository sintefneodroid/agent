#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .architecture import Architecture

__author__ = 'cnheider'

from torch import nn
from torch.nn import functional as F, init

from utilities.initialisation import fan_in_init


class ActorArchitecture(Architecture):

  def __init__(self, input_size, hidden_layers, output_size, output_activation):
    '''
Initialize a Actor for low dimensional environment.
    num_feature: number of features of input.
    num_action: number of available actions in the environment.
'''
    super().__init__()

    self._input_size = input_size
    self._hidden_layers = hidden_layers
    self._output_size = output_size
    self.activation = output_activation

    self.fc1 = nn.Linear(self._input_size[0], self._hidden_layers[0])
    fan_in_init(self.fc1.weight)

    self.fc2 = nn.Linear(self._hidden_layers[0], self._hidden_layers[1])
    fan_in_init(self.fc2.weight)

    self.head = nn.Linear(self._hidden_layers[1], self._output_size[0])
    init.uniform_(self.head.weight, -3e-3, 3e-3)
    init.uniform_(self.head.bias, -3e-3, 3e-3)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.tanh(self.head(x))
    if self.activation:
      x = self.activation(x, -1)
    return x
