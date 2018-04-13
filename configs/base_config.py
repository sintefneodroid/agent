#!/usr/bin/env python3
# coding=utf-8
__author__='cnheider'
"""
Description: Config for training
Author: Christian Heider Nielsen
"""

from pathlib import Path

import torch.nn.functional as F

from utilities import *

CONFIG_NAME = __name__
CONFIG_FILE = __file__
USE_LOGGING = True

# CUDA
USE_CUDA_IF_AVAILABLE = True
if USE_CUDA_IF_AVAILABLE:
  USE_CUDA_IF_AVAILABLE = torch.cuda.is_available()

DoubleTensor = torch.cuda.DoubleTensor if USE_CUDA_IF_AVAILABLE else torch.DoubleTensor
FloatTensor = torch.cuda.FloatTensor if USE_CUDA_IF_AVAILABLE else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA_IF_AVAILABLE else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA_IF_AVAILABLE else torch.ByteTensor

# CONSTANTS
MOVING_AVERAGE_WINDOW = 100
SPACER_SIZE = 60
RANDOM_SEED = 6
SAVE_MODEL_INTERVAL = 100

# Visualisation
USE_VISDOM = False
START_VISDOM_SERVER = False
VISDOM_SERVER = 'http://localhost'
# if not START_VISDOM_SERVER:
#  VISDOM_SERVER = 'http://visdom.ml'

# Paths
PROJECT = 'Neodroid'
# DATA_SET_DIRECTORY = os.path.join('/home/heider/Datasets', PROJECT)
# TARGET_FILE_NAME = 'target_position_rotation.csv'
# DEPTH_IMAGES_DIRECTORY = 'depth'


PROJECT_DIRECTORY = Path('/home/heider/Github/Neodroid/agent')

MODEL_DIRECTORY = PROJECT_DIRECTORY / 'models'
CONFIG_DIRECTORY = PROJECT_DIRECTORY / 'configs'
LOG_DIRECTORY = PROJECT_DIRECTORY / 'logs'

# Environment parameters
CONNECT_TO_RUNNING = True
RENDER_ENVIRONMENT = False
ENVIRONMENT_NAME = 'grid_world'
SOLVED_REWARD = 0.9
EPISODES = 4000
MAX_ROLLOUT_LENGTH = 1000
ACTION_MAGNITUDES = 10000
STATE_TENSOR_TYPE = FloatTensor
VALUE_TENSOR_TYPE = FloatTensor
ACTION_TENSOR_TYPE = LongTensor

# Exploration epsilon random action parameters
EXPLORATION_EPSILON_START = 0.9
EXPLORATION_EPSILON_END = 0.05
EXPLORATION_EPSILON_DECAY = 35000

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

EVALUATION_FUNCTION = F.smooth_l1_loss

# Optimiser
OPTIMISER_TYPE = torch.optim.Adam
# OPTIMISER_TYPE = torch.optim.RMSprop
OPTIMISER_LEARNING_RATE = 0.0025
OPTIMISER_WEIGHT_DECAY = 1e-5
OPTIMISER_ALPHA = 0.9
OPTIMISER_EPSILON = 1e-02
OPTIMISER_MOMENTUM = 0.0

# Architecture
ARCH_PARAMS = {
  'input_size':    '',  # Obtain from environment
  'hidden_layers': [32, 16],
  'output_size':   '',  # Obtain from environment
  'use_bias':      False
  }
ARCH = CategoricalMLP
