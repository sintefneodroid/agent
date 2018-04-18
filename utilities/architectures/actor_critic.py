#!/usr/bin/env python3
# coding=utf-8
from utilities.architectures.architecture import Architecture

__author__ = 'cnheider'

from torch import nn
from torch.nn import functional as F


class ActorCriticNetwork(Architecture):
  """
  An actor-critic network that shared lower-layer representations but
  have distinct output layers
  """

  def __init__(self,
               input_size,
               hidden_size,
               actor_hidden_size,
               actor_output_size,
               critic_hidden_size,
               critic_output_size,
               actor_output_activation,
               continuous):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size

    self.actor_hidden_size = actor_hidden_size
    self.actor_output_size = actor_output_size

    self.critic_hidden_size = critic_hidden_size
    self.critic_output_size = critic_output_size

    self.continuous = continuous

    self.actor_output_activation = actor_output_activation

    self.fc1 = nn.Linear(self.input_size, self.hidden_size[0])
    self.fc2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])

    self.actor_fc1 = nn.Linear(self.hidden_size[1], self.actor_hidden_size[0])
    self.actor_head = nn.Linear(self.actor_hidden_size[0], self.actor_output_size[0])

    self.critic_fc1 = nn.Linear(self.hidden_size[1], self.critic_hidden_size[0])
    self.critic_output = nn.Linear(self.critic_hidden_size[0], self.critic_output_size[0])

  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))

    prob = self.actor_fc1(x)
    # act = self.actor_output_activation(act)
    prob = self.actor_head(prob)
    prob_out = F.softmax(prob, dim=1)

    value = self.critic_fc1(x)
    value_out = self.critic_output(value)

    if not self.continuous:
      return prob_out, value_out
    else:
      log_prob = F.log_softmax(prob, dim=1)
      return prob_out, log_prob, value_out
