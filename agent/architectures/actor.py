#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn

from agent.utilities import init_weights
from .mlp import MLP

__author__ = 'cnheider'

from torch.nn import init


class DDPGActorArchitecture(MLP):

  def __init__(self, output_activation: callable = torch.tanh, init_weight=3e-3, **kwargs):

    super().__init__(**kwargs)

    self._output_activation = output_activation

    low, high = -init_weight, init_weight
    init.uniform_(self._head.weight, low, high)
    init.uniform_(self._head.bias, low, high)

  def forward(self, x, **kwargs):
    val = super().forward(x)

    if self._output_activation:
      val = self._output_activation(val)

    return val


class ContinuousActorArchitecture(MLP):

  def __init__(self, *, output_size, std=0.0, **kwargs):
    super().__init__(output_size=output_size, **kwargs)

    self._log_std = torch.nn.Parameter(torch.ones(1, output_size[0]) * std)
    self.apply(init_weights)

  def forward(self, x, **kwargs):
    mean = super().forward(x)  # .view(-1,1)
    std = self._log_std.exp()  # .expand_as(mean)
    return mean, std
