#!/usr/bin/env python3
# coding=utf-8
"""
Description: Config for training
Author: Christian Heider Nielsen
"""

from pathlib import Path

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
RANDOM_SEED = 8
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
ENVIRONMENT = 'grid_world'
SOLVED_REWARD = 0.9
NUM_EPISODES = 1000
MAX_ROLLOUT_LENGTH = 1000
ACTION_MAGNITUDES = 10000
STATE_TENSOR_TYPE = FloatTensor
VALUE_TENSOR_TYPE = FloatTensor
ACTION_TENSOR_TYPE = LongTensor

# Epsilon random action parameters
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 35000

# Training parameters
LOAD_PREVIOUS_MODEL_IF_AVAILABLE = False
DOUBLE_DQN = True
SIGNAL_CLIPPING = False
CLAMP_GRADIENT = True
BATCH_SIZE = 32
LEARNING_FREQUENCY = 4
SYNC_TARGET_MODEL_FREQUENCY = 10000
REPLAY_MEMORY_SIZE = 1000000
INITIAL_OBSERVATION_PERIOD = 50000
DISCOUNT_FACTOR = 0.99
UPDATE_DIFFICULTY_INTERVAL = 1000

# Optimiser
OPTIMISER = torch.optim.Adam
# OPTIMISER = torch.optim.RMSprop
LEARNING_RATE = 0.00025
WEIGHT_DECAY = 1e-5
ALPHA = 0.95
EPSILON = 0.01

# Curriculum
random_motion_horizon = 40
CANDIDATES_SIZE = 3
CANDIDATE_ROLLOUTS = 3

low = 0.1
high = 0.9

CURRICULUM = {
  'level1': {
    'when_reward':   0.95,
    'configurables': {
      'Difficulty': 1,
      }
    },
  'level2': {
    'when_reward':   0.95,
    'configurables': {
      'Difficulty': 2,
      }
    },
  'level3': {
    'when_reward':   0.95,
    'configurables': {
      'Difficulty': 3,
      }
    }
  }

CURRICULUM2 = {
  'level1': {
    'when_reward':   0.5,
    'configurables': {
      'WallColorVariation':  [0.0, 0.0, 0.0],
      'StartBoundaryRadius': 1,
      }
    },
  'level2': {
    'when_reward':   0.7,
    'configurables': {
      'WallColorVariation':  [0.1, 0.1, 0.1],
      'StartBoundaryRadius': 2,
      }
    },
  'level3': {
    'when_reward':   0.8,
    'configurables': {
      'WallColorVariation':  [0.5, 0.5, 0.5],
      'StartBoundaryRadius': 3,
      }
    }
  }

# Architecture
POLICY_ARCH_PARAMS = {
  'input_size':    '',  # Obtain from environment
  'activation':    F.leaky_relu,
  'hidden_layers': [128, 64, 32, 16],
  'output_size':   '',  # Obtain from environment,
  'use_bias':      False
  }
POLICY_ARCH = CategoricalMLP

VALUE_ARCH_PARAMS = {
  'input_size':    '',  # Obtain from environment
  'activation':    F.relu,
  'hidden_layers': [128, 64, 32, 16],
  'output_size':   '',  # Obtain from environment
  'use_bias':      False
  }
VALUE_ARCH = MLP
