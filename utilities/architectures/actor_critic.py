#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'

from torch import nn
from torch.nn import functional as F


class ActorCriticNetwork(nn.Module):
  """
  An actor-critic network that shared lower-layer representations but
  have distinct output layers
  """

  def __init__(self, C):
    super(ActorCriticNetwork, self).__init__()
    self.fc1 = nn.Linear(C.ARCH_PARAMS['input_size'][0], C.ARCH_PARAMS['actor_hidden_size'][0])
    self.fc2 = nn.Linear(C.ARCH_PARAMS['actor_hidden_size'][0], C.ARCH_PARAMS['actor_hidden_size'][0])
    self.actor_linear = nn.Linear(C.ARCH_PARAMS['actor_hidden_size'][0], C.ARCH_PARAMS['output_size'][0])
    self.critic = nn.Linear(C.ARCH_PARAMS['actor_hidden_size'][0], C.ARCH_PARAMS['critic_output_size'][0])
    self.actor_output_activation = C.ARCH_PARAMS['actor_output_activation']

    self.continuous = C.ARCH_PARAMS['continuous']

  def __call__(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))

    prob = self.actor_linear(x)
    # act = self.actor_output_activation(act)
    prob = F.softmax(prob, dim=1)

    value = self.critic(x)
    if not self.continuous:
      return prob, value
    else:
      log_prob = F.log_softmax(x, dim=1)
      return prob, log_prob, value
