#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroidagent.architectures import CategoricalCNN

from neodroidagent.utilities import ReplayBuffer

__author__ = "Christian Heider Nielsen"
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
DISCOUNT_FACTOR = 0.95
RENDER_ENVIRONMENT = False
DOUBLE_DQN = True
SYNC_TARGET_MODEL_FREQUENCY = 1000
CONNECT_TO_RUNNING = True

# EVALUATION_FUNCTION = lambda Q_state, Q_true_state: (Q_state - Q_true_state).pow(2).mean()

ARCH = CategoricalCNN
OPTIMISER_TYPE = torch.optim.Adam
OPTIMISER_LEARNING_RATE = 0.0025
# ENVIRONMENT_NAME = 'CartPole-v0'
ENVIRONMENT_NAME = "grd_camera"
# 'LunarLander-v2' #(coord_x, coord_y, vel_x, vel_y, angle,
# angular_vel, l_leg_on_ground, r_leg_on_ground)


POLICY_ARCH_PARAMS = {
    "input_shape": None,  # Obtain from environment
    "hidden_layer_activation": F.leaky_relu,
    "hidden_layers": [128, 64, 32, 16],
    "output_shape": None,  # Obtain from environment
    "use_bias": True,
}
POLICY_ARCH = CategoricalCNN
