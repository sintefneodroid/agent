#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Christian Heider Nielsen"

from neodroidagent.common import MultiVariateNormalMLP

"""
Description: Config for training
Author: Christian Heider Nielsen
"""

CONFIG_NAME = __name__
from pathlib import Path

CONFIG_FILE_PATH = Path(__file__)

ROLLOUTS = 10000
INITIAL_OBSERVATION_PERIOD = 0
LEARNING_FREQUENCY = 1
REPLAY_MEMORY_SIZE = 10000
MEMORY = ReplayBuffer(REPLAY_MEMORY_SIZE)

BATCH_SIZE = 128
DISCOUNT_FACTOR = 0.99
RENDER_ENVIRONMENT = False
DOUBLE_DQN = True
SYNC_TARGET_MODEL_FREQUENCY = 1000
CONNECT_TO_RUNNING = True

# EVALUATION_FUNCTION = lambda Q_state, Q_true_state: (Q_state - Q_true_state).pow(2).mean()

POLICY_ARCH = MultiVariateNormalMLP
OPTIMISER_TYPE = torch.optim.RMSprop  # torch.optim.Adam
# ENVIRONMENT_NAME = 'CartPole-v0'
ENVIRONMENT_NAME = "c2d"
# 'LunarLander-v2' #(coord_x, coord_y, vel_x, vel_y, angle,
# angular_vel, l_leg_on_ground, r_leg_on_ground)


# Architecture
POLICY_ARCH_PARAMS = {
    "input_shape": None,  # Obtain from environment
    "hidden_layers": None,
    "output_shape": None,  # Obtain from environment
    "hidden_layer_activation": torch.tanh,
    "use_bias": True,
}
