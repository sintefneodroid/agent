#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .mlp import MLP

__author__ = 'cnheider'
import torch
from torch import nn
from torch.nn import init


class DDPGCriticArchitecture(MLP):

  def __init__(self,
               *,
               init_weight=3e-3,
               **kwargs):
    super().__init__(**kwargs)

    prev_layer_size = self._input_shape

    if len(self._hidden_layers) > 1:
      prev_layer_size = self._hidden_layers[-2]

    setattr(self,
            f'_fc{self.num_of_layer}',
            nn.Linear(prev_layer_size + self._output_shape,
                      self._hidden_layers[-1]))

    self._head = nn.Linear(self._hidden_layers[-1], 1)

    low, high = -init_weight, init_weight
    init.uniform_(self._head.weight, low, high)
    init.uniform_(self._head.bias, low, high)

  def forward(self,
              x,
              *,
              actions,
              **kwargs):
    # assert isinstance(x,Tensor)
    # assert isinstance(actions,Tensor)

    for i in range(1, self.num_of_layer):  # Not top-inclusive
      layer = getattr(self, f'_fc{i}')
      x = layer(x)
      x = self._hidden_layer_activation(x)

    last_h_layer = getattr(self, f'_fc{self.num_of_layer}')
    x = torch.cat((x, actions), 1)
    x = last_h_layer(x)
    x = self._hidden_layer_activation(x)

    x = self._head(x)

    return x
