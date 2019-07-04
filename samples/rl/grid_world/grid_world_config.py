#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from agent.agents.pg_agent import PGAgent
from agent.architectures import MLP
from agent.memory import ReplayBuffer

__author__ = 'cnheider'
'''
Description: Config for training
Author: Christian Heider Nielsen
'''
from agent.configs.base_config import *

CONFIG_NAME = __name__
CONFIG_FILE = __file__

ROLLOUTS = 1000
INITIAL_OBSERVATION_PERIOD = 0
LEARNING_FREQUENCY = 1
REPLAY_MEMORY_SIZE = 10000
MEMORY = ReplayBuffer(REPLAY_MEMORY_SIZE)

EXPLORATION_SPEC = ExplorationSpecification(0, 0, 0)

AGENT_SPEC = GDCS(PGAgent, {})

BATCH_SIZE = 128
DISCOUNT_FACTOR = 0.95
RENDER_ENVIRONMENT = False
SIGNAL_CLIPPING = True
DOUBLE_DQN = False
SYNC_TARGET_MODEL_FREQUENCY = 1000
CONNECT_TO_RUNNING = True

# EVALUATION_FUNCTION = lambda Q_state, Q_true_state: (Q_state - Q_true_state).pow(2).mean()

VALUE_ARCH = MLP
OPTIMISER_TYPE = torch.optim.Adam
OPTIMISER_LEARNING_RATE = 0.1
# ENVIRONMENT_NAME = 'CartPole-v0'
ENVIRONMENT_NAME = 'grd'
# 'LunarLander-v2' #(coord_x, coord_y, vel_x, vel_y, angle,
# angular_vel, l_leg_on_ground, r_leg_on_ground)
