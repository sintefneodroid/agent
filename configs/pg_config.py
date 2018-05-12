#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'

from configs.base_config import *

CONFIG_NAME = __name__
CONFIG_FILE = __file__

ENVIRONMENT_NAME = 'CartPole-v0'
# ENVIRONMENT_NAME = 'LunarLander-v2' # (coord_x, coord_y, vel_x, vel_y, angle, angular_vel, l_leg_on_ground,
# r_leg_on_ground)
# ENVIRONMENT_NAME = 'small_grid_world'

CONNECT_TO_RUNNING = False
RENDER_ENVIRONMENT = False

ROLLOUTS = 2000

EVALUATION_FUNCTION = torch.nn.CrossEntropyLoss

DISCOUNT_FACTOR = 0.99
OPTIMISER_LEARNING_RATE = 1e-4
PG_ENTROPY_REG = 1e-4

# Architecture
POLICY_ARCH_PARAMS = {
  'input_size': None,  # Obtain from environment
  'activation': F.leaky_relu,
  'hidden_size':[128, 64, 32, 16],
  'output_size':None,  # Obtain from environment
  'use_bias':   True,
  }
POLICY_ARCH = CategoricalMLP
