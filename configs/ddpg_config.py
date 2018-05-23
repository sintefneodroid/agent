#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
from agents.ddpg_agent import DDPGAgent

__author__ = 'cnheider'
'''
Description: Config for training
Author: Christian Heider Nielsen
'''

# General
from configs.base_config import *
from utilities.random_process.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess

CONFIG_NAME = __name__
CONFIG_FILE = __file__

AGENT_TYPE = DDPGAgent

# Optimiser
OPTIMISER_TYPE = torch.optim.Adam
OPTIMISER_LEARNING_RATE = 0.00025
OPTIMISER_WEIGHT_DECAY = 1e-5
OPTIMISER_ALPHA = 0.95
DISCOUNT_FACTOR = 0.99
TARGET_UPDATE_TAU = 0.001

STATE_TENSOR_TYPE = torch.float
VALUE_TENSOR_TYPE = torch.float
ACTION_TENSOR_TYPE = torch.float

EVALUATION_FUNCTION = F.smooth_l1_loss

ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
Q_WEIGHT_DECAY = 0.01

ACTOR_OPTIMISER_SPEC = U.OptimiserSpecification(
    constructor=OPTIMISER_TYPE, kwargs=dict(lr=ACTOR_LEARNING_RATE)
    )

CRITIC_OPTIMISER_SPEC = U.OptimiserSpecification(
    constructor=OPTIMISER_TYPE,
    kwargs=dict(lr=CRITIC_LEARNING_RATE, weight_decay=Q_WEIGHT_DECAY),
    )

RANDOM_PROCESS_THETA = 0.15
RANDOM_PROCESS_SIGMA = 0.2
RANDOM_PROCESS = OrnsteinUhlenbeckProcess(
    theta=RANDOM_PROCESS_THETA, sigma=RANDOM_PROCESS_SIGMA
    )

MEMORY = U.TransitionBuffer(REPLAY_MEMORY_SIZE)

ACTION_CLIPPING = False
SIGNAL_CLIPPING = False

ENVIRONMENT_NAME = 'Pendulum-v0'
# ENVIRONMENT_NAME = 'MountainCarContinuous-v0'
# ENVIRONMENT_NAME = 'InvertedPendulum-v2'
# ENVIRONMENT_NAME = 'Reacher-v1'
# ENVIRONMENT_NAME = 'Hopper-v1'
# ENVIRONMENT_NAME = 'Ant-v1'
# ENVIRONMENT_NAME = 'Humanoid-v1'
# ENVIRONMENT_NAME = 'HalfCheetah-v1'

# Architecture
ACTOR_ARCH_PARAMS = {
  'input_size':       None,  # Obtain from environment
  'hidden_size':      [128, 64],
  'output_activation':None,
  'output_size':      None,  # Obtain from environment
  }
ACTOR_ARCH = U.ActorArchitecture

CRITIC_ARCH_PARAMS = {
  'input_size':       None,  # Obtain from environment
  'hidden_size':      [128, 64],
  'output_activation':None,
  'output_size':      None,  # Obtain from environment
  }
CRITIC_ARCH = U.CriticArchitecture
