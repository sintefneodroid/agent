#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import pathlib
import time

import torch

from draugr.torch_utilities import global_torch_device
from neodroidagent import PROJECT_NAME
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
USE_LOGGING = True

# Environment Related Parameters
ENVIRONMENT_NAME = "ConnectToRunning"
CONNECT_TO_RUNNING = False
RENDER_ENVIRONMENT = False
RUN_TRAINING = True
CONTINUE = True

# Training parameters
LOAD_PREVIOUS_MODEL_IF_AVAILABLE = False
SIGNAL_CLIPPING = False
GRADIENT_CLIPPING = False

DISCOUNT_FACTOR = 0.999
UPDATE_DIFFICULTY_INTERVAL = 1000
ITERATIONS = 4000

STATE_TYPE = torch.float
VALUE_TYPE = torch.float
ACTION_TYPE = torch.long

# Optimiser
OPTIMISER_SPEC = GDKC(torch.optim.Adam, lr=1e-3)
SCHEDULER_SPEC = GDKC(
    torch.optim.lr_scheduler.StepLR, step_size=int(math.sqrt(ITERATIONS)), gamma=0.65
)

# CUDA
USE_CUDA = True
DEVICE = global_torch_device(USE_CUDA)

# CONSTANTS
MOVING_AVERAGE_WINDOW = 100
SPACER_SIZE = 60
SAVE_MODEL_INTERVAL = 100
SEED = 2 ** 12 - 1  # Upto power 32

if __name__ == "__main__":
    print(CONFIG_FILE_PATH)
