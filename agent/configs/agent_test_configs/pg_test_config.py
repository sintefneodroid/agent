#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from agent.architectures import MLP

from .base_test_config import *

__author__ = 'cnheider'

CONFIG_NAME = __name__
CONFIG_FILE = __file__

CONNECT_TO_RUNNING = False
RENDER_ENVIRONMENT = True

EVALUATION_FUNCTION = torch.nn.CrossEntropyLoss

DISCOUNT_FACTOR = 0.98
PG_ENTROPY_REG = 3e-3

# Architecture
POLICY_ARCH_SPEC = GDCS(MLP, NOD(**{
  'input_shape':            None,  # Obtain from environment
  'hidden_layer_activation':torch.tanh,
  'hidden_layers':          None,
  'output_shape':           None,  # Obtain from environment
  'use_bias':               True,
  }))
