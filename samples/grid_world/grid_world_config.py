#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from draugr.torch_utilities import MLP
from neodroidagent.agents.torch_agents.model_free import PolicyGradientAgent
from warg import GDKC

__author__ = "Christian Heider Nielsen"

from neodroidagent.utilities import ExplorationSpecification

"""
Description: Config for training
Author: Christian Heider Nielsen
"""

CONFIG_NAME = __name__
from pathlib import Path
from neodroidagent.utilities import ReplayBuffer

CONFIG_FILE_PATH = Path(__file__)

ROLLOUTS = 1000
INITIAL_OBSERVATION_PERIOD = 0
LEARNING_FREQUENCY = 1
REPLAY_MEMORY_SIZE = 10000
MEMORY = ReplayBuffer(REPLAY_MEMORY_SIZE)

EXPLORATION_SPEC = ExplorationSpecification(0, 0, 0)

AGENT_SPEC = GDKC(PolicyGradientAgent, {})

BATCH_SIZE = 128
DISCOUNT_FACTOR = 0.999
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
