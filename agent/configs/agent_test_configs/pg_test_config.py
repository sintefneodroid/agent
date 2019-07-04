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

ENVIRONMENT_NAME = 'CartPole-v1'
MODEL_DIRECTORY = PROJECT_APP_PATH.user_data / ENVIRONMENT_NAME / LOAD_TIME / 'models'
CONFIG_DIRECTORY = PROJECT_APP_PATH.user_data / ENVIRONMENT_NAME / LOAD_TIME / 'configs'
LOG_DIRECTORY = PROJECT_APP_PATH.user_log / ENVIRONMENT_NAME / LOAD_TIME

DISCOUNT_FACTOR = 0.95
PG_ENTROPY_REG = 3e-3

# Architecture
POLICY_ARCH_SPEC = GDCS(MLP, NOD(**{
  'input_shape':            None,  # Obtain from environment
  'hidden_layer_activation':torch.relu,
  'hidden_layers':          None,
  'output_shape':           None,  # Obtain from environment
  'use_bias':               True,
  }))
