#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroidagent.architectures import MLP

__author__ = "Christian Heider Nielsen"
"""
Description: Config for training
Author: Christian Heider Nielsen
"""

from neodroidagent.configs.base_config import *

CONFIG_NAME = __name__
import pathlib

CONFIG_FILE_PATH = pathlib.Path(__file__)

# Paths
AGENT_TYPE_NAME = "DefaultAgent"
MODEL_DIRECTORY = (
    PROJECT_APP_PATH.user_data
    / ENVIRONMENT_NAME
    / AGENT_TYPE_NAME
    / LOAD_TIME
    / "models"
)
CONFIG_DIRECTORY = (
    PROJECT_APP_PATH.user_data
    / ENVIRONMENT_NAME
    / AGENT_TYPE_NAME
    / LOAD_TIME
    / "configs"
)
LOG_DIRECTORY = (
    PROJECT_APP_PATH.user_log / ENVIRONMENT_NAME / AGENT_TYPE_NAME / LOAD_TIME
)

# Architecture
POLICY_ARCH_SPEC = GDKC(
    CategoricalMLP,
    NOD(
        **{
            "input_shape": None,  # Obtain from environment
            "hidden_layer_activation": torch.tanh,
            "hidden_layers": [128, 64, 32, 16],
            "output_shape": None,  # Obtain from environment,
            "use_bias": False,
        }
    ),
)

VALUE_ARCH_PARAMS = GDKC(
    MLP,
    NOD(
        **{
            "input_shape": None,  # Obtain from environment
            "hidden_layer_activation": torch.tanh,
            "hidden_layers": [128, 64, 32, 16],
            "output_shape": None,  # Obtain from environment
            "use_bias": False,
        }
    ),
)

# Optimiser
OPTIMISER_SPEC = GDKC(
    torch.optim.Adam, NOD(lr=0.00025, weight_decay=1e-5, alpha=0.95, epsilon=0.01)
)

# Curriculum
RANDOM_MOTION_HORIZON = 20
CANDIDATE_SET_SIZE = 3
CANDIDATE_ROLLOUTS = 3

LOW = 0.1
HIGH = 0.9
