#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from typing import Sequence

import numpy
from numpy import prod

from neodroidagent.interfaces.architecture import Architecture
from neodroidagent.utilities import to_tensor, xavier_init
from warg.named_ordered_dictionary import NOD

__author__ = 'cnheider'

'''
Description: Multi Layer Perceptron
Author: Christian Heider Nielsen
'''
import torch
from torch import nn


class MLP(Architecture):
  '''
OOOO input_shape
|XX|                                        fc1
OOOO hidden_layer_size * (Weights,Biases)
|XX|                                        fc2
OOOO hidden_layer_size * (Weights,Biases)
|XX|                                        fc3
0000 output_shape * (Weights,Biases)
'''

  def __init__(self,
               *,
               input_shape: Sequence = (10,),
               hidden_layers: Sequence = None,
               hidden_layer_activation: callable = torch.relu,
               output_shape: Sequence = (2,),
               use_bias: bool = True,
               auto_build_hidden_layers_if_none=True,
               input_multiplier=32,
               output_multiplier=16,
               **kwargs
               ):
    super().__init__(**kwargs)

    assert input_shape is not None
    assert output_shape is not None

    self._input_shape = None
    self._output_shape = None

    self.infer_input_shape(input_shape)
    self.infer_output_shape(output_shape)

    if not hidden_layers and auto_build_hidden_layers_if_none:
      hidden_layers = self.construct_progressive_hidden_layers(self._input_shape,
                                                               self._output_shape,
                                                               input_multiplier,
                                                               output_multiplier)
    elif not isinstance(hidden_layers, Sequence):
      hidden_layers = (hidden_layers,)

    self._hidden_layers = hidden_layers
    self._hidden_layer_activation = hidden_layer_activation
    self._use_bias = use_bias
    previous_layer_size = self._hidden_layers[0] * self._input_shape[0]
    self.num_of_layer = len(self._hidden_layers)

    for i in range(1, self._input_shape[0] + 1):
      layer = nn.Linear(self._input_shape[1],
                        self._hidden_layers[0],
                        bias=self._use_bias)
      setattr(self, f'_in{i}', layer)

    if self.num_of_layer > 0:
      for i in range(2, self.num_of_layer + 1):
        layer = nn.Linear(previous_layer_size,
                          self._hidden_layers[i - 1],
                          bias=self._use_bias)
        setattr(self, f'_fc{i}', layer)
        previous_layer_size = self._hidden_layers[i - 1]

    for i in range(1, self._output_shape[0] + 1):
      layer = nn.Linear(previous_layer_size,
                        self._output_shape[1],
                        bias=self._use_bias)
      setattr(self, f'_out{i}', layer)

    xavier_init(self)

  @staticmethod
  def construct_progressive_hidden_layers(_input_shape, _output_shape, input_multiplier, output_multiplier):
    h_1_size = int(sum(_input_shape) * input_multiplier)
    h_3_size = int(sum(_output_shape) * output_multiplier)

    h_2_size = int(numpy.sqrt(h_1_size * h_3_size))

    hidden_layers = NOD(h_1_size,
                        h_2_size,
                        h_3_size
                        ).as_list()
    return hidden_layers

  def infer_input_shape(self, input_shape):
    if isinstance(input_shape, Sequence):
      assert len(input_shape) > 0, f'Got length {len(input_shape)}'
      if len(input_shape) > 2:
        # self._input_shape = functools.reduce(operator.mul,input_shape)
        self._input_shape = input_shape[0], prod(input_shape[1:])
        logging.info(
          f'Flattening input {input_shape} to flattened vectorised input shape {self._input_shape}')
      elif len(input_shape) < 2:
        self._input_shape = (1, input_shape[0])
        logging.info(f'Inflating input shape {input_shape} to vectorised input shape {self._input_shape}')
      else:
        self._input_shape = input_shape
    elif isinstance(input_shape, int):
      self._input_shape = (1, input_shape)
      logging.info(f'Inflating input shape {input_shape} to vectorised input shape {self._input_shape}')
    else:
      raise ValueError(f'Can not use {input_shape} as input shape')

  def infer_output_shape(self, output_shape):
    if isinstance(output_shape, Sequence):
      assert len(output_shape) > 0, f'Got length {len(output_shape)}'
      if len(output_shape) > 2:
        self._output_shape = output_shape[0], prod(output_shape[1:])
        logging.info(
          f'Flattening output shape {output_shape} to flattened vectorised output shape {self._output_shape}')
      elif len(output_shape) < 2:
        self._output_shape = (1, output_shape[0])
        logging.info(f'Inflating output shape {output_shape} to vectorised output shape {self._output_shape}')
      else:
        self._output_shape = output_shape
    elif isinstance(output_shape, int):
      self._output_shape = (1, output_shape)
      logging.info(f'Inflating output shape {output_shape} to vectorised output shape {self._output_shape}')
    else:
      raise ValueError(f'Can not use {output_shape} as output shape')

  def forward(self, *x, **kwargs):
    '''

    :param x:
    :return output:
    '''

    x_len = len(x)
    if x_len != self.input_shape[0]:
      raise ValueError(f'{self.input_shape[0]} input arguments expected, {len(x)} was supplied')

    ins = []
    for i in range(1, self._input_shape[0] + 1):
      layer = getattr(self, f'_in{i}')

      x_s = x[i - 1]
      ins.append(layer(x_s))

    val = torch.cat(ins, dim=-1)

    for i in range(2, self.num_of_layer + 1):
      layer = getattr(self, f'_fc{i}')
      val = layer(val)
      val = self._hidden_layer_activation(val)

    val_dis = val
    outs = []
    for i in range(1, self._output_shape[0] + 1):
      layer = getattr(self, f'_out{i}')
      a = layer(val_dis)
      outs.append(a)

    return outs

  def __repr__(self):
    num_trainable_params = sum(p.numel()
                               for p in self.parameters()
                               if p.requires_grad)
    num_params = sum(param.numel() for param in self.parameters())

    return f'{super().__repr__()}\ntrainable/num_params: {num_trainable_params}/{num_params}\n'


class SingleHeadMLP(MLP):

  def forward(self, *x, **kwargs):
    outs = super().forward(*x, **kwargs)
    return outs[0]


if __name__ == '__main__':

  def test_single_dim():
    pos_size = (4,)
    a_size = (1,)
    model = MLP(input_shape=pos_size, output_shape=a_size)

    pos_1 = to_tensor(numpy.random.rand(64, pos_size[0]))
    print(model(pos_1))


  def test_hidden_dim():
    pos_size = (4,)
    hidden_size = (2, 3)
    a_size = (2,)
    model = MLP(input_shape=pos_size, hidden_layers=hidden_size, output_shape=a_size)

    pos_1 = to_tensor(numpy.random.rand(64, pos_size[0]))
    print(model(pos_1))


  def test_multi_dim():
    pos_size = (2, 3, 2)
    a_size = (2, 4, 5)
    model = MLP(input_shape=pos_size, output_shape=a_size)

    pos_1 = to_tensor(numpy.random.rand(64, prod(pos_size[1:])))
    pos_2 = to_tensor(numpy.random.rand(64, prod(pos_size[1:])))
    print(model(pos_1, pos_2))


  test_single_dim()
  test_hidden_dim()
  test_multi_dim()
