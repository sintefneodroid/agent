#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from agent.architectures import MLP, ContinuousActorArchitecture
from torch.nn import MSELoss
from .base_test_config import *

__author__ = 'cnheider'
'''
Description: Config for training
Author: Christian Heider Nielsen
'''

# General

CONFIG_NAME = __name__
CONFIG_FILE = __file__

ENVIRONMENT_NAME = 'Pendulum-v0'
CONNECT_TO_RUNNING =False
RENDER_ENVIRONMENT = True
TEST_INTERVAL = 1000

# Optimiser
OPTIMISER_TYPE = torch.optim.Adam

INITIAL_OBSERVATION_PERIOD = 0

STEPS = 20

MEMORY_CAPACITY = STEPS
BATCH_SIZE = 64

TARGET_UPDATE_INTERVAL = 1000
TARGET_UPDATE_TAU = 1.0
MAX_GRADIENT_NORM = None

GAE_TAU = 0.95
DISCOUNT_FACTOR = 0.99

REACHED_HORIZON_PENALTY = -10.

# CRITIC_LOSS = F.smooth_l1_loss
CRITIC_LOSS = MSELoss

EXPLORATION_EPSILON_START = 0.99
EXPLORATION_EPSILON_END = 0.05
EXPLORATION_EPSILON_DECAY = 500

PPO_EPOCHS = 4

SEED = 66

VALUE_REG_COEF = 0.5
ENTROPY_REG_COEF = 1.0

LR_FUNC = lambda a:OPTIMISER_LEARNING_RATE * (1. - a)

SURROGATE_CLIPPING_VALUE = 0.2  # initial probability ratio clipping range
SURROGATE_CLIP_FUNC = lambda a:SURROGATE_CLIPPING_VALUE * (
    1. - a
)  # clip range schedule function

ACTOR_LR = 3e-3
# Architecture
ACTOR_ARCH_PARAMETERS = NOD(**{
  'input_size':             None,  # Obtain from environment
  'hidden_layers':          [256],
  'hidden_layer_activation':torch.relu,
  'output_size':            None,  # Obtain from environment
  })
ACTOR_ARCH = ContinuousActorArchitecture

CRITIC_LR = 3e-3
CRITIC_ARCH_PARAMETERS = NOD(**{
  'input_size':             None,  # Obtain from environment
  'hidden_layers':          [256],
  'hidden_layer_activation':torch.relu,
  'output_size':            None,  # Obtain from environment
  })
CRITIC_ARCH = MLP

ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 3e-3
Q_WEIGHT_DECAY = 3e-2

ACTOR_OPTIMISER_SPEC = U.OptimiserSpecification(constructor=OPTIMISER_TYPE,
                                                kwargs=dict(lr=ACTOR_LEARNING_RATE)
                                                )

CRITIC_OPTIMISER_SPEC = U.OptimiserSpecification(constructor=OPTIMISER_TYPE,
                                                 kwargs=dict(lr=CRITIC_LEARNING_RATE,
                                                             weight_decay=Q_WEIGHT_DECAY),
                                                 )
