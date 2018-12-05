#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .mlp import MLP

__author__ = 'cnheider'
import torch
from torch import nn, Tensor
from torch.nn import init


class DDPGCriticArchitecture(MLP):

  def __init__(self,
               *,
               output_size,
               init_weight=3e-3,
               **kwargs):
    super().__init__(output_size=output_size, **kwargs)

    setattr(self,
            f'_fc{self.num_of_layer}',
            nn.Linear(self._hidden_layers[-2] + output_size[0],
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
    assert type(x) is Tensor
    assert type(actions) is Tensor

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
