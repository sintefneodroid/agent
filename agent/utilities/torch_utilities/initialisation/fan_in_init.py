#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn

__author__ = 'cnheider'

import numpy as np
from torch.nn import init


def fan_in_init(tensor):
  fan_in = tensor.size(1)
  v = 1.0 / np.sqrt(fan_in)
  init.uniform_(tensor, -v, v)


def xavier_init(model, activation='relu'):
  gain = nn.init.calculate_gain(activation)
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
      nn.init.xavier_uniform_(m.weight,
                              gain=gain)
