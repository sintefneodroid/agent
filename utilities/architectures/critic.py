#!/usr/bin/env python3
# coding=utf-8
import torch
from torch import nn
from torch.nn import functional as F, init

from utilities.initialisation.fan_in_init import fan_in_init


class CriticArchitecture(nn.Module):
  def __init__(self, ARCH_PARAMS):
    """
    Initialize a Critic for low dimensional environment.
        num_feature: number of features of input.

    """
    super().__init__()
    self.fc1 = nn.Linear(ARCH_PARAMS['input_size'][0], ARCH_PARAMS['hidden_size'][0])
    fan_in_init(self.fc1.weight)

    self.fc2 = nn.Linear(ARCH_PARAMS['hidden_size'][0] + ARCH_PARAMS['output_size'][0],
                         ARCH_PARAMS['hidden_size'][
                           0])  # Actions are not included until the 2nd layer of Q.
    fan_in_init(self.fc2.weight)

    self.fc3 = nn.Linear(ARCH_PARAMS['hidden_size'][0], 1)
    init.uniform(self.fc3.weight, -3e-3, 3e-3)
    init.uniform(self.fc3.bias, -3e-3, 3e-3)

  def forward(self, states, actions):
    x = F.relu(self.fc1(states))
    x = torch.cat((x, actions), 1)
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
