#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'
'''
Description: Config for training
Author: Christian Heider Nielsen
'''

# General
from configs.base_config import *

CONFIG_NAME = __name__
CONFIG_FILE = __file__

# Optimiser
OPTIMISER_TYPE = torch.optim.Adam
ENVIRONMENT_NAME = 'Pendulum-v0'
# ENVIRONMENT_NAME = 'InvertedDoublePendulum-v2'
# ENVIRONMENT_NAME = 'Reacher-v2'
# ENVIRONMENT_NAME = 'PongNoFrameskip-v4'

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
CRITIC_LOSS = nn.MSELoss

EXPLORATION_EPSILON_START = 0.99
EXPLORATION_EPSILON_END = 0.05
EXPLORATION_EPSILON_DECAY = 500

SEED = 66

VALUE_REG_COEF = 1.
ENTROPY_REG_COEF = 0.1

LR_FUNC = lambda a:OPTIMISER_LEARNING_RATE * (1. - a)

SURROGATE_CLIP = 0.2  # initial probability ratio clipping range
SURROGATE_CLIP_FUNC = lambda a:SURROGATE_CLIP * (
    1. - a
)  # clip range schedule function

ACTOR_CRITIC_LR = 3e-4

ACTOR_CRITIC_ARCH_PARAMS = {
  'input_size':             None,
  'hidden_size':            [32, 32],
  'actor_hidden_size':      [32],
  'critic_hidden_size':     [32],
  'actor_output_size':      None,
  'actor_output_activation':F.log_softmax,
  'critic_output_size':     [1],
  'continuous':             True,
  }
