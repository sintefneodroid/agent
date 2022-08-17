#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"
"""
Description: Config for training
Author: Christian Heider Nielsen
"""


import time
from os import cpu_count
from pathlib import Path

from draugr.torch_utilities import global_torch_device

from neodroidagent import PROJECT_NAME
from neodroidagent.agents.agent import TogglableLowHigh, TogglableValue
from trolls.render_mode import RenderModeEnum


PROJECT_NAME = PROJECT_NAME
CONFIG_NAME = __name__
CONFIG_FILE_PATH = Path(__file__)
CONFIG_AUTHOR = __author__
LOAD_TIME = str(int(time.time()))

VERBOSE = False
USE_LOGGING = False

# Environment Related Parameters
ENVIRONMENT_NAME = "ConnectToRunning"
CONNECT_TO_RUNNING = False
RENDER_ENVIRONMENT = False
RENDER_FREQUENCY = 0
RENDER_MODE = RenderModeEnum.rgb_array
NUM_ENVS = cpu_count()

# Training parameters
LOAD_PREVIOUS_MODEL_IF_AVAILABLE = False

# Clipping
SIGNAL_CLIPPING = TogglableLowHigh(False, -1.0, 1.0)
ACTION_CLIPPING = TogglableLowHigh(False, -1.0, 1.0)
GRADIENT_CLIPPING = TogglableLowHigh(False, -1.0, 1.0)
GRADIENT_NORM_CLIPPING = TogglableValue(False, 1.0)

DISCOUNT_FACTOR = 0.99  # For sparse signal settings is it very important to keep the long term signals relevant by making them stretch far back in the rollout trace

ITERATIONS = 4000

# CUDA
USE_CUDA = True
GLOBAL_DEVICE = global_torch_device(USE_CUDA)

# CONSTANTS
MOVING_AVERAGE_WINDOW = 100
SPACER_SIZE = 60
SAVE_MODEL_INTERVAL = 100
SEED = 2**6 - 1  # Up to power 32

if __name__ == "__main__":
    print(CONFIG_FILE_PATH)
