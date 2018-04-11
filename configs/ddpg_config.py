#!/usr/bin/env python3
# coding=utf-8
"""
Description: Config for training
Author: Christian Heider Nielsen
"""

import utilities as U
# General
from configs.base_config import *

# Optimiser
OPTIMISER_TYPE = torch.optim.Adam
LEARNING_RATE = 0.00025
WEIGHT_DECAY = 1e-5
ALPHA = 0.95
EPSILON = 0.01
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
Q_WEIGHT_DECAY = 0.01
TAU = 0.001
THETA = 0.15
SIGMA = 0.2
LOG_EVERY_N_EPS = 10
ACTION_CLIPPING = False
SIGNAL_CLIPPING = False

ENVIRONMENT_NAME = 'satellite_test'

# Architecture
ACTOR_ARCH_PARAMS = {
  'input_size':        '',  # Obtain from environment
  'hidden_size':       [128, 64, 32, 16],
  'output_activation': None,
  'output_size':       ''  # Obtain from environment
  }
ACTOR_ARCH = U.ActorArchitecture

CRITIC_ARCH_PARAMS = {
  'input_size':        '',  # Obtain from environment
  'hidden_size':       [128, 64, 32, 16],
  'output_activation': None,
  'output_size':       ''  # Obtain from environment
  }
CRITIC_ARCH = U.CriticArchitecture
