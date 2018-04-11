#!/usr/bin/env python3
# coding=utf-8

from torch import nn
from torch.nn import functional as F


class ActorCriticNetwork(nn.Module):
  """
  An actor-critic network that shared lower-layer representations but
  have distinct output layers
  """

  def __init__(self, C, critic_output_size=1):
    super(ActorCriticNetwork, self).__init__()
    self.fc1 = nn.Linear(C.ARCH_PARAMS['input_size'][0], C.ARCH_PARAMS['actor_hidden_size'][0])
    self.fc2 = nn.Linear(C.ARCH_PARAMS['actor_hidden_size'][0], C.ARCH_PARAMS['actor_hidden_size'][0])
    self.actor_linear = nn.Linear(C.ARCH_PARAMS['actor_hidden_size'][0], C.ARCH_PARAMS['output_size'][0])
    self.critic_linear = nn.Linear(C.ARCH_PARAMS['actor_hidden_size'][0], critic_output_size)
    self.actor_output_activation = C.ARCH_PARAMS['actor_output_activation']

  def __call__(self, state):
    out = F.relu(self.fc1(state))
    out = F.relu(self.fc2(out))

    act = self.actor_linear(out)
    # act = self.actor_output_activation(act)
    act = F.softmax(act, dim=1)

    val = self.critic_linear(out)

    return act, val
