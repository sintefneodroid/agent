#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from agent.architectures import MLP

__author__ = 'cnheider'
'''
Description: Config for training
Author: Christian Heider Nielsen
'''

from agent.configs.base_config import *

CONFIG_NAME = __name__
CONFIG_FILE = __file__

# Architecture
POLICY_ARCH_SPEC = GDCS(CategoricalMLP, NOD(**{
  'input_shape':            None,  # Obtain from environment
  'hidden_layer_activation':torch.tanh,
  'hidden_layers':          [128, 64, 32, 16],
  'output_shape':           None,  # Obtain from environment,
  'use_bias':               False,
  }))

VALUE_ARCH_PARAMS = GDCS(MLP, NOD(**{
  'input_shape':            None,  # Obtain from environment
  'hidden_layer_activation':torch.tanh,
  'hidden_layers':          [128, 64, 32, 16],
  'output_shape':           None,  # Obtain from environment
  'use_bias':               False,
  }))

# Optimiser
OPTIMISER_SPEC = GDCS(torch.optim.Adam, NOD(lr=0.00025, weight_decay=1e-5, alpha=0.95, epsilon=0.01))

# Curriculum
RANDOM_MOTION_HORIZON = 20
CANDIDATE_SET_SIZE = 3
CANDIDATE_ROLLOUTS = 3

LOW = 0.1
HIGH = 0.9
