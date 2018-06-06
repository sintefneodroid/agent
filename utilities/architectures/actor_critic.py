#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

from utilities.architectures.architecture import Architecture

__author__ = 'cnheider'

from torch import nn
from torch.nn import functional as F


class ActorCriticNetwork(Architecture):
  '''
An actor-critic network that shared lower-layer representations but
have distinct output layers
'''

  def __init__(
      self,
      input_size,
      hidden_layers,
      actor_hidden_layers,
      actor_output_size,
      critic_hidden_layers,
      critic_output_size,
      actor_output_activation,
      continuous,
      ):
    super().__init__()

    self.input_size = input_size
    self.hidden_layers = hidden_layers

    self.actor_hidden_layers = actor_hidden_layers
    self.actor_output_size = actor_output_size

    self.critic_hidden_layers = critic_hidden_layers
    self.critic_output_size = critic_output_size

    self.continuous = continuous

    self.actor_output_activation = actor_output_activation

    self.fc1 = nn.Linear(self.input_size[0], self.hidden_layers[0])
    self.fc2 = nn.Linear(self.hidden_layers[0], self.hidden_layers[1])

    self.actor_fc1 = nn.Linear(self.hidden_layers[1], self.actor_hidden_layers[0])
    self.actor_head = nn.Linear(
        self.actor_hidden_layers[0], self.actor_output_size[0]
        )

    if self.continuous:
      self.log_std = nn.Parameter(torch.zeros(1, self.actor_output_size[0]))

    self.critic_fc1 = nn.Linear(self.hidden_layers[1], self.critic_hidden_layers[0])
    self.critic_output = nn.Linear(
        self.critic_hidden_layers[0], self.critic_output_size[0]
        )

  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))

    mu = self.actor_fc1(x)
    mu = self.actor_head(mu)
    if self.actor_output_activation:
      mu = self.actor_output_activation(mu, dim=1)

    value = self.critic_fc1(x)
    value = self.critic_output(value)

    if self.continuous:
      return mu, self.log_std, value
    else:
      return mu, value
