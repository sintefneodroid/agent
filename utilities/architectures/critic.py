#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
from utilities.architectures.architecture import Architecture

__author__ = 'cnheider'
import torch
from torch import nn
from torch.nn import functional as F, init

from utilities.initialisation.fan_in_init import fan_in_init


class CriticArchitecture(Architecture):

  def __init__(self, input_size, hidden_size, output_size, output_activation):
    '''
Initialize a Critic for low dimensional environment.
    num_feature: number of features of input.

'''
    super().__init__()

    self._input_size = input_size
    self._hidden_size = hidden_size
    self._output_size = output_size

    self.fc1 = nn.Linear(self._input_size[0], self._hidden_size[0])
    fan_in_init(self.fc1.weight)

    self.fc2 = nn.Linear(
        self._hidden_size[0] + self._output_size[0], self._hidden_size[1]
        )  # Actions are not included until the 2nd layer of Q.
    fan_in_init(self.fc2.weight)

    self.head = nn.Linear(self._hidden_size[1], 1)
    init.uniform_(self.head.weight, -3e-3, 3e-3)
    init.uniform_(self.head.bias, -3e-3, 3e-3)

  def forward(self, states, actions):
    x = F.relu(self.fc1(states))
    x = torch.cat((x, actions), 1)
    x = F.relu(self.fc2(x))
    x = self.head(x)
    return x
