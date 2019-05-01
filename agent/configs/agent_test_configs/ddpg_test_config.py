#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from agent.agents.ddpg_agent import DDPGAgent
from agent.architectures import DDPGActorArchitecture, DDPGCriticArchitecture
from .base_test_config import *

__author__ = 'cnheider'
'''
Description: Config for training
Author: Christian Heider Nielsen
'''

# General

CONFIG_NAME = __name__
CONFIG_FILE = __file__

AGENT_TYPE = DDPGAgent

CONNECT_TO_RUNNING=False
ENVIRONMENT_NAME = 'Pendulum-v0'

# Optimiser
OPTIMISER_TYPE = torch.optim.Adam
OPTIMISER_LEARNING_RATE = 0.00025
OPTIMISER_WEIGHT_DECAY = 1e-5
OPTIMISER_ALPHA = 0.95
DISCOUNT_FACTOR = 0.99
TARGET_UPDATE_TAU = 3e-3
RENDER_FREQUENCY = 5
RENDER_ENVIRONMENT = True


STATE_TYPE = torch.float
VALUE_TYPE = torch.float
ACTION_TYPE = torch.float

EVALUATION_FUNCTION = F.smooth_l1_loss

ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 3e-3
Q_WEIGHT_DECAY = 0.01
BATCH_SIZE = 64

ACTOR_OPTIMISER_SPEC = U.OptimiserSpecification(constructor=OPTIMISER_TYPE,
                                                kwargs=dict(lr=ACTOR_LEARNING_RATE)
                                                )

CRITIC_OPTIMISER_SPEC = U.OptimiserSpecification(constructor=OPTIMISER_TYPE,
                                                 kwargs=dict(lr=CRITIC_LEARNING_RATE,
                                                             weight_decay=Q_WEIGHT_DECAY),
                                                 )

RANDOM_PROCESS_THETA = 0.15
RANDOM_PROCESS_SIGMA = 0.2
RANDOM_PROCESS = U.OrnsteinUhlenbeckProcess(theta=RANDOM_PROCESS_THETA,
                                            sigma=RANDOM_PROCESS_SIGMA
                                            )

MEMORY = U.TransitionBuffer(REPLAY_MEMORY_SIZE)

ACTION_CLIPPING = False
SIGNAL_CLIPPING = False

ROLLOUTS = 1000

# Architecture
ACTOR_ARCH_PARAMETERS = NOD(**{
  'input_size':       None,  # Obtain from environment
  #'hidden_layers' : [256],
  'output_activation':torch.tanh,
  'output_size':      None,  # Obtain from environment
  })
ACTOR_ARCH = DDPGActorArchitecture

CRITIC_ARCH_PARAMETERS = NOD(**{
  'input_size':       None,  # Obtain from environment
  #'hidden_layers' : [256],
  'output_size':      None,  # Obtain from environment
  })
CRITIC_ARCH = DDPGCriticArchitecture
