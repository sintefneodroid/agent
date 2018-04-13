#!/usr/bin/env python3
# coding=utf-8
__author__='cnheider'

from torch import nn
from torch.nn import functional as F, init

from utilities.initialisation.fan_in_init import fan_in_init


class ActorArchitecture(nn.Module):
  def __init__(self, ARCH_PARAMS):
    """
    Initialize a Actor for low dimensional environment.
        num_feature: number of features of input.
        num_action: number of available actions in the environment.
    """
    super().__init__()

    self.fc1 = nn.Linear(ARCH_PARAMS['input_size'][0], ARCH_PARAMS['hidden_size'][0])
    fan_in_init(self.fc1.weight)

    self.fc2 = nn.Linear(ARCH_PARAMS['hidden_size'][0], ARCH_PARAMS['hidden_size'][0])
    fan_in_init(self.fc2.weight)

    self.fc3 = nn.Linear(ARCH_PARAMS['hidden_size'][0], ARCH_PARAMS['output_size'][0])
    init.uniform(self.fc3.weight, -3e-3, 3e-3)
    init.uniform(self.fc3.bias, -3e-3, 3e-3)

    self.activation = ARCH_PARAMS['output_activation']

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.tanh(self.fc3(x))
    if self.activation:
      x = self.activation(x, -1)
    return x
