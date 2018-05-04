#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'
'''
Description: Config for training
Author: Christian Heider Nielsen
'''

from pathlib import Path
from configs.base_config import *
from utilities import *

CONFIG_NAME = __name__
CONFIG_FILE = __file__

# Architecture
POLICY_ARCH_PARAMS = {
  'input_size':    None,  # Obtain from environment
  'activation':    F.leaky_relu,
  'hidden_size': [128, 64, 32, 16],
  'output_size':   None,  # Obtain from environment,
  'use_bias':      False,
  }
POLICY_ARCH = CategoricalMLP

VALUE_ARCH_PARAMS = {
  'input_size':    None,  # Obtain from environment
  'activation':    F.relu,
  'hidden_size': [128, 64, 32, 16],
  'output_size':   None,  # Obtain from environment
  'use_bias':      False,
  }
VALUE_ARCH = MLP

# Optimiser
OPTIMISER = torch.optim.Adam
# OPTIMISER = torch.optim.RMSprop
LEARNING_RATE = 0.00025
WEIGHT_DECAY = 1e-5
ALPHA = 0.95
EPSILON = 0.01

# Curriculum
RANDOM_MOTION_HORIZON = 40
CANDIDATES_SIZE = 3
CANDIDATE_ROLLOUTS = 3

LOW = 0.1
HIGH = 0.9

CURRICULUM = {
  'level1': {'when_reward': 0.95, 'configurables': {'Difficulty': 1}},
  'level2': {'when_reward': 0.95, 'configurables': {'Difficulty': 2}},
  'level3': {'when_reward': 0.95, 'configurables': {'Difficulty': 3}},
  }

CURRICULUM2 = {
  'level1': {
    'when_reward':   0.5,
    'configurables': {
      'WallColorVariation': [0.0, 0.0, 0.0], 'StartBoundaryRadius': 1
      },
    },
  'level2': {
    'when_reward':   0.7,
    'configurables': {
      'WallColorVariation': [0.1, 0.1, 0.1], 'StartBoundaryRadius': 2
      },
    },
  'level3': {
    'when_reward':   0.8,
    'configurables': {
      'WallColorVariation': [0.5, 0.5, 0.5], 'StartBoundaryRadius': 3
      },
    },
  }

