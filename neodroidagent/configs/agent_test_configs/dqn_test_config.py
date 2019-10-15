#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroidagent.architectures import MLP
from neodroidagent.memory import ReplayBuffer

from .base_dicrete_test_config import *

__author__ = 'Christian Heider Nielsen'
'''
Description: Config for training
Author: Christian Heider Nielsen
'''

CONFIG_NAME = __name__
import pathlib

CONFIG_FILE_PATH = pathlib.Path(__file__)

INITIAL_OBSERVATION_PERIOD = 0
LEARNING_FREQUENCY = 1
REPLAY_MEMORY_SIZE = 10000
MEMORY = ReplayBuffer(REPLAY_MEMORY_SIZE)
EXPLORATION_SPEC = ExplorationSpecification(0.99, 0.05, 10000)

BATCH_SIZE = 128
DISCOUNT_FACTOR = 0.95
RENDER_ENVIRONMENT = True
SIGNAL_CLIPPING = True
DOUBLE_DQN = True
SYNC_TARGET_MODEL_FREQUENCY = 1000

# EVALUATION_FUNCTION = lambda Q_state, Q_true_state: (Q_state - Q_true_state).pow(2).mean()

OPTIMISER_SPEC = GDKC(torch.optim.RMSprop, {})  # torch.optim.Adam

# Architecture
VALUE_ARCH_SPEC = GDKC(MLP,
                       NOD(**{'input_shape':            None,  # Obtain from environment
                              'hidden_layers':          None,
                              'output_shape':           None,  # Obtain from environment
                              'hidden_layer_activation':torch.relu,
                              'use_bias':               True,
                              }))
