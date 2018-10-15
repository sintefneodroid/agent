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

  def __init__(self,
               *,
               input_size:list=[10],
               hidden_layers:list =[32],
               output_size:list=[2],
               activation:callable=F.tanh,
               use_bias:bool=True,
               **kwargs
               ):
    super().__init__(**kwargs)

    self._input_size = input_size[0]
    self._hidden_layers = hidden_layers
    self._activation = activation
    self._output_size = output_size[0]
    self._use_bias = use_bias

    previous_layer_size = self._input_size

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
        previous_layer_size, self._output_size, bias=self._use_bias
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

  def __init__(self, *, heads_hidden_sizes=[32,64], heads=[2,1], **kwargs):
    super().__init__(**kwargs)

    assert len(heads_hidden_sizes) == len(heads)

    self._heads_hidden_sizes = heads_hidden_sizes
    self._heads = heads

    self.num_of_heads = len(self._heads)
    if self.num_of_heads > 0:
      for i in range(1,self.num_of_heads+ 1):
        head_hidden = nn.Linear(self._output_size, self._heads_hidden_sizes[i-1],bias=self._use_bias)
        setattr(self, f'subhead{str(i)}_hidden', head_hidden)
        head = nn.Linear(self._heads_hidden_sizes[i-1], self._heads[i-1],bias=self._use_bias)
        setattr(self, f'subhead{str(i)}', head)
    else:
      raise ValueError('Number of heads must be >0')

  def forward(self, x, **kwargs):
    x = super().forward(x, **kwargs)

    output = []
    for i in range(1, self.num_of_heads + 1):
      head_hidden = getattr(self, f'subhead{str(i)}_hidden')
      x_s = head_hidden(x)
      head = getattr(self, f'subhead{str(i)}')
      sub_res = head(x_s)

      #if type(sub_res) is not list:
      #  sub_res = [sub_res]

      output.append(sub_res)

    return output


class SingleDistributionMLP(MultiHeadedMLP):
  def __init__(self, **kwargs):
    heads = [1, 1]

    super().__init__(heads=heads, **kwargs)


class RecurrentCategoricalMLP(MLP):

  def __init__(self, r_hidden_layers=10, **kwargs):
    super().__init__(**kwargs)
    self._r_hidden_layers = r_hidden_layers
    self._r_input_size = self._output_size + r_hidden_layers

    self.hidden = nn.Linear(self._r_input_size, r_hidden_layers,bias=self._use_bias)
    self.out = nn.Linear(self._r_input_size, r_hidden_layers,bias=self._use_bias)

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
