#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.distributions import Categorical, Normal

from utilities.torch_utilities.initialisation import init_weights
from utilities.architectures.mlp import MultiHeadedMLP

__author__ = 'cnheider'

from torch import nn


class ActorCriticNetwork(MultiHeadedMLP):
  '''
An actor-critic network that shared lower-layer representations but
have distinct output layers
'''

  def __init__(
      self,
      *,
      head_size,
      distribution=Categorical,
      std=0.0,
      **kwargs
      ):
    assert len(head_size) == 2

    super().__init__(heads=head_size, **kwargs)

    self.log_std = nn.Parameter(torch.ones(1, self._heads[0]) * std)
    self._distribution = distribution

    self.apply(init_weights)

  def forward(self, state, **kwargs):
    mu, value = super().forward(state, **kwargs)

    action_distribution = self.actor(mu)
    return action_distribution, value

  def actor(self, x):
    std = self.log_std.exp().expand_as(x)
    action_distribution = self._distribution(x, std)
    return action_distribution


class ActorCritic(nn.Module):
  def __init__(self, num_inputs, num_outputs, hidden_size, activation=torch.nn.ReLU(),
               distribution=Normal, std=0.0):
    super(ActorCritic, self).__init__()

    self.common = nn.Linear(num_inputs, hidden_size * 2)

    self.critic = nn.Sequential(
        nn.Linear(hidden_size * 2, hidden_size),
        activation,
        nn.Linear(hidden_size, 1)
        )

    self.actor = nn.Sequential(
        nn.Linear(hidden_size * 2, hidden_size),
        activation,
        nn.Linear(hidden_size, num_outputs),
        )
    self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
    self._distribution = distribution

    self.apply(init_weights)

  def forward(self, x):
    x = self.common(x)
    value = self.critic(x)
    mu = self.actor(x)
    std = self.log_std.exp().expand_as(mu)
    dist = self._distribution(mu, std)
    return dist, value
