#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroidagent.architectures import CNN
from neodroidagent.configs.agent_test_configs.base_dicrete_test_config import *
from warg import GDKC

from neodroidagent.utilities import ReplayBuffer

__author__ = "Christian Heider Nielsen"
"""
Description: Config for training
Author: Christian Heider Nielsen
"""

CONFIG_NAME = __name__
from pathlib import Path

CONFIG_FILE_PATH = Path(__file__)

# Exploration
EXPLORATION_EPSILON_START = 1.0
EXPLORATION_EPSILON_END = 0.04
EXPLORATION_EPSILON_DECAY = 400

ROLLOUTS = 10000
INITIAL_OBSERVATION_PERIOD = 0
LEARNING_FREQUENCY = 1
REPLAY_MEMORY_SIZE = 10000
MEMORY = ReplayBuffer(REPLAY_MEMORY_SIZE)

BATCH_SIZE = 128
DISCOUNT_FACTOR = 0.95
RENDER_ENVIRONMENT = False
DOUBLE_DQN = True
SYNC_TARGET_MODEL_FREQUENCY = 1000

# EVALUATION_FUNCTION = lambda Q_state, Q_true_state: (Q_state - Q_true_state).pow(2).mean()

OPTIMISER_SPEC = GDKC(torch.optim.RMSprop, {})  # torch.optim.Adam

# Architecture
VALUE_ARCH_SPEC = GDKC(
    CNN,
    NOD(
        input_shape=None,  # Obtain from environment
        input_channels=None,
        hidden_layers=[64, 32, 16],
        output_shape=None,  # Obtain from environment
        output_channels=None,
        hidden_layer_activation=F.relu,
        use_bias=True,
    ),
)
