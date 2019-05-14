#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from warg import NOD

from agent.agents.pg_agent import PGAgent
from agent.architectures import CategoricalMLP
from agent.utilities.specifications.exploration_specification import ExplorationSpecification
from agent.utilities.specifications.generalised_delayed_construction_specification import GDCS

__author__ = 'cnheider'
'''
Description: Config for training
Author: Christian Heider Nielsen
'''

PROJECT = 'Neodroid'
CONFIG_NAME = __name__
CONFIG_FILE = __file__
VERBOSE = False
USE_LOGGING = True

# class LearningConfig(object):
#  pass

# class EnvironmentConfig(object):
#  pass

input_size = None  # Obtain from environment
hidden_layers = None  # Obtain from input and output size
output_size = None  # Obtain from environment

# Architecture
POLICY_ARCH_SPEC = GDCS(CategoricalMLP, NOD(
    input_size=input_size,
    hidden_layers=hidden_layers,
    output_size=output_size,
    hidden_layer_activation=torch.relu,
    use_bias=True
    ))

# Environment Related Parameters
CONNECT_TO_RUNNING = False
RENDER_ENVIRONMENT = False
ENVIRONMENT_NAME = 'grd'
SOLVED_REWARD = 0.9
ACTION_MAGNITUDES = 10000

# Epsilon Exploration
EXPLORATION_SPEC = ExplorationSpecification(0.99, 0.05, 10000)

# Training parameters
LOAD_PREVIOUS_MODEL_IF_AVAILABLE = False
DOUBLE_DQN = False
SIGNAL_CLIPPING = False
CLAMP_GRADIENT = False
BATCH_SIZE = 32
LEARNING_FREQUENCY = 4
SYNC_TARGET_MODEL_FREQUENCY = 10000
REPLAY_MEMORY_SIZE = 1000000
INITIAL_OBSERVATION_PERIOD = 10000
DISCOUNT_FACTOR = 0.99
UPDATE_DIFFICULTY_INTERVAL = 1000
ROLLOUTS = 4000
STATE_TYPE = torch.float
VALUE_TYPE = torch.float
ACTION_TYPE = torch.long
EVALUATION_FUNCTION = F.smooth_l1_loss

# Optimiser
OPTIMISER_SPEC = GDCS(torch.optim.Adam, NOD(
    lr=0.0025,
    weight_decay=1e-5,
    eps=1e-02))

# Paths
# PROJECT_DIRECTORY = Path.cwd()
PROJECT_DIRECTORY = Path.home() / 'Models' / 'Neodroid' / str(int(time.time()))
MODEL_DIRECTORY = PROJECT_DIRECTORY / 'models'
CONFIG_DIRECTORY = PROJECT_DIRECTORY / 'configs'
LOG_DIRECTORY = PROJECT_DIRECTORY / 'logs'

# CUDA
USE_CUDA = True
if USE_CUDA:  # If available
  USE_CUDA = torch.cuda.is_available()

# Visualisation
USE_VISDOM = False
START_VISDOM_SERVER = False
VISDOM_SERVER = 'http://localhost'
if not START_VISDOM_SERVER:
  # noinspection PyRedeclaration
  VISDOM_SERVER = 'http://visdom.ml'

# CONSTANTS
MOVING_AVERAGE_WINDOW = 100
SPACER_SIZE = 60
SEED = 6
SAVE_MODEL_INTERVAL = 100
