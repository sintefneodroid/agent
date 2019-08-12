#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn

__author__ = 'cnheider'

import numpy as np
from torch.nn import init, Module


def fan_in_init(model: Module):
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
      fan_in = m.weight.size(1)
      v = 1.0 / np.sqrt(fan_in)
      init.uniform_(m.weight, -v, v)


def xavier_init(model: Module, activation='relu'):
  gain = nn.init.calculate_gain(activation)
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
      nn.init.xavier_uniform_(m.weight,
                              gain=gain)
