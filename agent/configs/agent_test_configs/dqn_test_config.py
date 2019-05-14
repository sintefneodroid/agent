#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from agent.architectures import MLP
from agent.utilities import ReplayBuffer
from .base_test_config import *

__author__ = 'cnheider'
'''
Description: Config for training
Author: Christian Heider Nielsen
'''

CONFIG_NAME = __name__
CONFIG_FILE = __file__

INITIAL_OBSERVATION_PERIOD = 0
LEARNING_FREQUENCY = 1
REPLAY_MEMORY_SIZE = 10000
MEMORY = ReplayBuffer(REPLAY_MEMORY_SIZE)

BATCH_SIZE = 128
DISCOUNT_FACTOR = 0.999
RENDER_ENVIRONMENT = True
SIGNAL_CLIPPING = True
DOUBLE_DQN = True
SYNC_TARGET_MODEL_FREQUENCY = 1000

# EVALUATION_FUNCTION = lambda Q_state, Q_true_state: (Q_state - Q_true_state).pow(2).mean()

OPTIMISER_SPEC = GDCS(torch.optim.RMSprop, {})  # torch.optim.Adam

# Architecture
VALUE_ARCH_SPEC = GDCS(MLP, NOD(**{
  'input_size':             None,  # Obtain from environment
  'hidden_layers':          None,
  'output_size':            None,  # Obtain from environment
  'hidden_layer_activation':torch.relu,
  'use_bias':               True,
  }))
