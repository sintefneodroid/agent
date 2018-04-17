#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'
"""
Description: Config for training
Author: Christian Heider Nielsen
"""

# General
from configs.base_config import *

CONFIG_NAME = __name__
CONFIG_FILE = __file__

# Optimiser
OPTIMISER_TYPE = torch.optim.Adam
# ENVIRONMENT_NAME = 'InvertedDoublePendulum-v1'
ENVIRONMENT_NAME = 'CartPole-v0'

EPISODES = 10000
EPISODES_BEFORE_TRAIN = 0

STEPS = 10

MEMORY_CAPACITY = STEPS
BATCH_SIZE = STEPS

TARGET_UPDATE_STEPS = 1000
TARGET_TAU = 1.0
MAX_GRADIENT_NORM = None

GAE_LAMBDA_PARAMETER = 0.95
GAMMA = 0.99

DONE_PENALTY = -10.
ROLLOUTS = int(10e6)

# CRITIC_LOSS = F.smooth_l1_loss
CRITIC_LOSS = nn.MSELoss

EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 500

SEED = 66

VALUE_REG_COEF = 1.
ENTROPY_REG_COEF = 0.1

LR_FUNC = lambda a: OPTIMISER_LEARNING_RATE * (1. - a)

CLIP = 0.2  # initial probability ratio clipping range
CLIP_FUNC = lambda a: CLIP * (1. - a)  # clip range schedule function

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4

USE_CUDA = False

ARCH_PARAMS = {'input_size':              '',
               'actor_hidden_size':       [32],
               'critic_hidden_size':      [32],
               'output_size':             '',
               'actor_output_activation': F.log_softmax,
               'critic_output_size':      [1],
               'continuous':              False
               }
