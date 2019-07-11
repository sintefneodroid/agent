#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections.abc import Collection

from agent.interfaces.architecture import Architecture

__author__ = 'cnheider'
from torch import nn
from torch.nn import functional as F

from agent.utilities import atari_initializer, ortho_weights


class CNN(Architecture):

  def __init__(self,
               *,
               input_shape,
               hidden_layers: Collection,
               output_shape,
               activation: callable = F.relu,
               **kwargs):
    super().__init__(**kwargs)

    self._input_channels = input_shape
    self._output_shape = output_shape
    self._activation = activation

    self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
    self.bn3 = nn.BatchNorm2d(32)
    self.head = nn.Linear(448, 2)

  def forward(self, x):
    x = self._activation(self.bn1(self.conv1(x)))
    x = self._activation(self.bn2(self.conv2(x)))
    x = self._activation(self.bn3(self.conv3(x)))
    return self.head(x.view(x.size(0), -1))


class CategoricalCNN(CNN):

  def forward(self, x):
    x = super().forward(x)
    return F.softmax(x, dim=0)


class AtariCNN(nn.Module):

  def __init__(self, num_actions):
    ''' Basic convolutional actor-critic network for Atari 2600 games

Equivalent to the network in the original DQN paper.

Args:
    num_actions (int): the number of available discrete actions
'''
    super().__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(4, 32, 8, stride=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(inplace=True),
        )

    self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(inplace=True))

    self.pi = nn.Linear(512, num_actions)
    self.v = nn.Linear(512, 1)

    self.num_actions = num_actions

    # parameter initialization
    self.apply(atari_initializer)
    self.pi.weight.data = ortho_weights(self.pi.weight.size(), scale=.01)
    self.v.weight.data = ortho_weights(self.v.weight.size())

  def forward(self, conv_in):
    ''' Module forward pass

Args:
    conv_in (Variable): convolutional input, shaped [N x 4 x 84 x 84]

Returns:
    pi (Variable): action probability logits, shaped [N x self.num_actions]
    v (Variable): value predictions, shaped [N x 1]
'''
    N = conv_in.size()[0]

    conv_out = self.conv(conv_in).view(N, 64 * 7 * 7)

    fc_out = self.fc(conv_out)

    pi_out = self.pi(fc_out)
    v_out = self.v(fc_out)

    return pi_out, v_out
