#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .architecture import Architecture

__author__ = 'cnheider'

'''
Description: Multi Layer Perceptron
Author: Christian Heider Nielsen
'''
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class MLP(Architecture):
  '''
OOOO input_size
|XX|                                        fc1
OOOO hidden_layer_size * (Weights,Biases)
|XX|                                        fc2
OOOO hidden_layer_size * (Weights,Biases)
|XX|                                        fc3
0000 output_size * (Weights,Biases)
'''

  def __init__(self, input_size, hidden_layers, output_size, activation, use_bias):
    super().__init__()

    self._input_size = input_size
    self._hidden_layers = hidden_layers
    self._activation = activation
    self._output_size = output_size
    self._use_bias = use_bias

    previous_layer_size = self._input_size[0]

    self.num_of_layer = len(self._hidden_layers)
    if self.num_of_layer > 0:
      for i in range(1, self.num_of_layer + 1):
        layer = nn.Linear(
            previous_layer_size, self._hidden_layers[i - 1], bias=self._use_bias
            )
        # fan_in_init(layer.weight)
        setattr(self, f'fc{i}', layer)
        previous_layer_size = self._hidden_layers[i - 1]

    self.head = nn.Linear(
        previous_layer_size, self._output_size[0], bias=self._use_bias
        )

  def forward(self, x, **kwargs):
    '''

:param x:
:return output:
'''
    assert type(x) is Tensor

    # if hasattr(self, 'num_of_layer'): # Safer but slower
    #  for i in range(1, self.num_of_layer + 1):
    #    if hasattr(self, 'fc' + str(i)):
    #      layer = getattr(self, 'fc' + str(i))
    #      x = F.relu(layer(x))

    for i in range(1, self.num_of_layer + 1):
      layer = getattr(self, f'fc{i}')
      x = layer(x)
      x = self._activation(x)

    return self.head(x)


class CategoricalMLP(MLP):

  def forward(self, x, **kwargs):
    x = super().forward(x, **kwargs)
    return F.softmax(x, dim=1)


class MultiHeadedMLP(MLP):

  def __init__(self, *, heads, **kwargs):
    super().__init__(**kwargs)

    self._heads = heads

    self.num_of_heads = len(self._heads)
    if self.num_of_heads > 0:
      for i in range(self.num_of_heads):
        head = nn.Linear(self._output_size[0], self._heads[i])
        # fan_in_init(layer.weight)
        setattr(self, f'subhead{str(i + 1)}', head)
    else:
      raise ValueError('Number of head must be >0')

  def forward(self, x, **kwargs):
    x = super().forward(x, **kwargs)

    output = []
    for i in range(1, self.num_of_heads + 1):
      head = getattr(self, 'subhead' + str(i))
      sub_res = head(x)
      if type(sub_res) is not list:
        sub_res = [sub_res]
      output.append(sub_res)

    return output


class DistributionMLP(MultiHeadedMLP):
  def __init__(self, **kwargs):
    heads = [1, 1]

    super().__init__(heads=heads, **kwargs)


class RecurrentCategoricalMLP(MLP):

  def __init__(self, r_hidden_layers=10, **kwargs):
    super().__init__(**kwargs)
    self._r_hidden_layers = r_hidden_layers
    self._r_input_size = self._output_size[0] + r_hidden_layers

    self.hidden = nn.Linear(self._r_input_size, r_hidden_layers)
    self.out = nn.Linear(self._r_input_size, r_hidden_layers)

    self._prev_hidden_x = torch.zeros(r_hidden_layers)

  def forward(self, x, **kwargs):
    x = super().forward(x, **kwargs)
    combined = torch.cat((x, self._prev_hidden_x), 1)
    out_x = self.out(combined)
    hidden_x = self.hidden(combined)
    self._prev_hidden_x = hidden_x

    return F.softmax(out_x, dim=1)


class ExposedRecurrentCategoricalMLP(RecurrentCategoricalMLP):

  def forward(self, x, hidden_x, **kwargs):
    self._prev_hidden_x = hidden_x
    out_x = super().forward(x, **kwargs)

    return F.softmax(out_x, dim=1), self._prev_hidden_x
