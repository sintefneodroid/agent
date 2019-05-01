#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
import random

from .atari_weight_init import *
from .fan_in_init import *
from .ortho_weight_init import *

import torch.nn as nn


def set_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  # neo.seed(seed)


def set_lr(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.normal_(m.weight, mean=0., std=0.1)
    nn.init.constant_(m.bias, 0.1)
