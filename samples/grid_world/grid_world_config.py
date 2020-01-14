#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroidagent.agents.torch_agents.model_free import PGAgent
from neodroidagent.architectures import MLP
from neodroidagent.memory import ReplayBuffer

__author__ = "Christian Heider Nielsen"
"""
Description: Config for training
Author: Christian Heider Nielsen
"""
from neodroidagent.configs.base_config import *

CONFIG_NAME = __name__
import pathlib

CONFIG_FILE_PATH = pathlib.Path(__file__)

ROLLOUTS = 1000
INITIAL_OBSERVATION_PERIOD = 0
LEARNING_FREQUENCY = 1
REPLAY_MEMORY_SIZE = 10000
MEMORY = ReplayBuffer(REPLAY_MEMORY_SIZE)

EXPLORATION_SPEC = ExplorationSpecification(0, 0, 0)

AGENT_SPEC = GDKC(PGAgent, {})

BATCH_SIZE = 128
DISCOUNT_FACTOR = 0.95
RENDER_ENVIRONMENT = False
DOUBLE_DQN = False
SYNC_TARGET_MODEL_FREQUENCY = 1000
CONNECT_TO_RUNNING = True

# EVALUATION_FUNCTION = lambda Q_state, Q_true_state: (Q_state - Q_true_state).pow(2).mean()

VALUE_ARCH = MLP
OPTIMISER_TYPE = torch.optim.Adam
OPTIMISER_LEARNING_RATE = 0.1
# ENVIRONMENT_NAME = 'CartPole-v0'
ENVIRONMENT_NAME = "grd"
# 'LunarLander-v2' #(coord_x, coord_y, vel_x, vel_y, angle,
# angular_vel, l_leg_on_ground, r_leg_on_ground)
