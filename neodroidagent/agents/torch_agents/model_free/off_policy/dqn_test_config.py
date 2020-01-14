#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroidagent.common.architectures.mlp import SingleHeadMLP
from neodroidagent.common.memory import ReplayBuffer

from neodroidagent.common.configs.base_dicrete_test_config import *

__author__ = "Christian Heider Nielsen"

from neodroidagent.utilities.exploration.exploration_specification import (
    ExplorationSpecification,
)

"""
Description: Config for training
Author: Christian Heider Nielsen
"""

CONFIG_NAME = __name__
import pathlib

CONFIG_FILE_PATH = pathlib.Path(__file__)

REPLAY_MEMORY_SIZE = 10000
MEMORY = ReplayBuffer(REPLAY_MEMORY_SIZE)
DOUBLE_DQN = False
LEARNING_FREQUENCY = 4
EXPLORATION_SPEC = ExplorationSpecification(0.995, 0.05, 10000)

BATCH_SIZE = 128
DISCOUNT_FACTOR = 0.95
RENDER_ENVIRONMENT = True
SYNC_TARGET_MODEL_FREQUENCY = 100
INITIAL_OBSERVATION_PERIOD = 1000

OPTIMISER_SPEC = GDKC(torch.optim.RMSprop)

# Architecture
VALUE_ARCH_SPEC = GDKC(
    SingleHeadMLP,
    hidden_layers=(24),
    hidden_layer_activation=torch.relu,
    use_dropout=False,
)
