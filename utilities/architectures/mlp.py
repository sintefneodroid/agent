#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'

"""
Description: Multi Layer Perceptron
Author: Christian Heider Nielsen
"""
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class MLP(nn.Module):
  """
  OOOO input_size
  |XX|                                        fc1
  OOOO hidden_layer_size * (Weights,Biases)
  |XX|                                        fc2
  OOOO hidden_layer_size * (Weights,Biases)
  |XX|                                        fc3
  0000 output_size * (Weights,Biases)
  """

  def __init__(self, **kwargs):
    super().__init__()

    self.input_size = kwargs['input_size'][0]
    self.hidden_sizes = kwargs['hidden_layers']
    self.activation = kwargs['activation']
    self.output_size = kwargs['output_size'][0]
    self.use_bias = kwargs['use_bias']

    previous_layer_size = self.input_size

    self.num_of_layer = len(self.hidden_sizes)
    if self.num_of_layer > 0:
      for i in range(self.num_of_layer):
        layer = nn.Linear(previous_layer_size, self.hidden_sizes[i], bias=self.use_bias)
        # fan_in_init(layer.weight)
        setattr(self, 'fc' + str(i + 1), layer)
        previous_layer_size = self.hidden_sizes[i]

    self.head = nn.Linear(previous_layer_size, self.output_size, bias=self.use_bias)

  def forward(self, x, **kwargs):
    """

    :param x:
    :return output:
    """
    assert type(x) is Variable

    # if hasattr(self, 'num_of_layer'): # Safer but slower
    #  for i in range(1, self.num_of_layer + 1):
    #    if hasattr(self, 'fc' + str(i)):
    #      layer = getattr(self, 'fc' + str(i))
    #      x = F.relu(layer(x))

    for i in range(1, self.num_of_layer + 1):
      layer = getattr(self, 'fc' + str(i))
      x = layer(x)
      x = self.activation(x)

    return self.head(x)


class CategoricalMLP(MLP):

  def forward(self, x, **kwargs):
    x = super().forward(x)
    return F.softmax(x, dim=1)


class MultiheadedMLP(MLP):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.heads = kwargs['heads']

    self.num_of_heads = len(self.heads)
    if self.num_of_heads > 0:
      for i in range(self.num_of_heads):
        layer = nn.Linear(self.output_size, self.heads[i])
        # fan_in_init(layer.weight)
        setattr(self, 'subhead' + str(i + 1), layer)
    else:
      raise ValueError('Number of head must be >0')

  def forward(self, x, **kwargs):
    x = super().forward(x)

    output = []
    for i in range(1, self.heads + 1):
      layer = getattr(self, 'subhead' + str(i))
      output.append(layer(x))

    return output


class RecurrentMLP(MLP):

  def __init__(self, r_hidden_size=10, **kwargs):
    super().__init__(**kwargs)
    self.hidden_size = r_hidden_size

    self.hidden = nn.Linear(kwargs['output_size'][0] + r_hidden_size, r_hidden_size)
    self.out = nn.Linear(kwargs['output_size'][0] + r_hidden_size, r_hidden_size)

  def forward(self, x, **kwargs):
    x = super().forward(x)
    combined = torch.cat((x, kwargs['hidden']), 1)
    out_x = self.out(combined)
    hidden_x = self.hidden(combined)

    return F.softmax(out_x, dim=1), hidden_x
