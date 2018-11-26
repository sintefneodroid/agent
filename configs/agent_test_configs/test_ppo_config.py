#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch.nn import MSELoss

__author__ = 'cnheider'
'''
Description: Config for training
Author: Christian Heider Nielsen
'''

# General
from configs.agent_test_configs.base_test_config import *

CONFIG_NAME = __name__
CONFIG_FILE = __file__

# Optimiser
OPTIMISER_TYPE = torch.optim.Adam


INITIAL_OBSERVATION_PERIOD = 0

STEPS = 10

MEMORY_CAPACITY = STEPS
BATCH_SIZE = STEPS

TARGET_UPDATE_INTERVAL = 1000
TARGET_UPDATE_TAU = 1.0
MAX_GRADIENT_NORM = None

GAE_TAU = 0.95
DISCOUNT_FACTOR = 0.99

REACHED_HORIZON_PENALTY = -10.
ROLLOUTS = int(10e6)

# CRITIC_LOSS = F.smooth_l1_loss
CRITIC_LOSS = MSELoss

EXPLORATION_EPSILON_START = 0.99
EXPLORATION_EPSILON_END = 0.05
EXPLORATION_EPSILON_DECAY = 500

SEED = 66

VALUE_REG_COEF = .5
ENTROPY_REG_COEF = 0.001

LR_FUNC = lambda a:OPTIMISER_LEARNING_RATE * (1. - a)

SURROGATE_CLIP = 0.2  # initial probability ratio clipping range
SURROGATE_CLIP_FUNC = lambda a:SURROGATE_CLIP * (
    1. - a
)  # clip range schedule function

ACTOR_CRITIC_LR = 3e-4
ACTOR_CRITIC_ARCH_PARAMS = {
  'input_size':   None,
  'hidden_layers':[32, 32],
  'output_size':  32,
  'heads':        [1, 1],
  'distribution': False
  }
