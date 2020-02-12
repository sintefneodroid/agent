#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import pathlib
import time

import torch

from draugr.torch_utilities import global_torch_device
from neodroidagent import PROJECT_NAME
from neodroidagent.agents.agent import ClipFeature
from warg.gdkc import GDKC

__author__ = "Christian Heider Nielsen"
"""
Description: Config for training
Author: Christian Heider Nielsen
"""

PROJECT_NAME = PROJECT_NAME
CONFIG_NAME = __name__
CONFIG_FILE_PATH = pathlib.Path(__file__)
CONFIG_AUTHOR = __author__
LOAD_TIME = str(int(time.time()))

VERBOSE = False
USE_LOGGING = False

# Environment Related Parameters
ENVIRONMENT_NAME = "ConnectToRunning"
CONNECT_TO_RUNNING = False
RENDER_ENVIRONMENT = False
# CONTINUE_TRAINING = False

# Training parameters
LOAD_PREVIOUS_MODEL_IF_AVAILABLE = False

# Clipping
SIGNAL_CLIPPING = ClipFeature(False, -1.0, 1.0)
ACTION_CLIPPING = ClipFeature(False, -1.0, 1.0)
GRADIENT_CLIPPING = ClipFeature(False, -1.0, 1.0)

DISCOUNT_FACTOR = 0.99
RENDER_FREQUENCY = 50
ITERATIONS = 4000

UPDATE_DIFFICULTY_INTERVAL = 1000

# CUDA
USE_CUDA = True
DEVICE = global_torch_device(USE_CUDA)

# CONSTANTS
MOVING_AVERAGE_WINDOW = 100
SPACER_SIZE = 60
SAVE_MODEL_INTERVAL = 100
SEED = 2 ** 6 - 1  # Up to power 32

if __name__ == "__main__":
    print(CONFIG_FILE_PATH)
