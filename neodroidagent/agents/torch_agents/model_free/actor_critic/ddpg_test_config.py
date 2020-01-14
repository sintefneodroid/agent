#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.nn.functional import smooth_l1_loss

from neodroidagent.common.architectures import SingleHeadMLP
from neodroidagent.common.architectures.experimental.merged import (
    SingleHeadMergedInputMLP,
)
from neodroidagent.common.memory import TransitionBuffer
from neodroidagent.utilities.exploration import OrnsteinUhlenbeckProcess
from neodroidagent.common.configs.base_continous_test_config import *

__author__ = "Christian Heider Nielsen"
"""
Description: Config for training
Author: Christian Heider Nielsen
"""

# General

CONFIG_NAME = __name__
import pathlib

CONFIG_FILE_PATH = pathlib.Path(__file__)

CONNECT_TO_RUNNING = False
ENVIRONMENT_NAME = "Pendulum-v0"

# Optimiser
TARGET_UPDATE_TAU = 3e-3
RENDER_FREQUENCY = 5
RENDER_ENVIRONMENT = True
REPLAY_MEMORY_SIZE = 10000

STATE_TYPE = torch.float
VALUE_TYPE = torch.float
ACTION_TYPE = torch.float

EVALUATION_FUNCTION = smooth_l1_loss

ACTOR_OPTIMISER_SPEC = GDKC(torch.optim.Adam, lr=0.00025)
CRITIC_OPTIMISER_SPEC = GDKC(torch.optim.Adam, lr=3e-3)

RANDOM_PROCESS_THETA = 0.15
RANDOM_PROCESS_SIGMA = 0.2
RANDOM_PROCESS = GDKC(
    OrnsteinUhlenbeckProcess, theta=RANDOM_PROCESS_THETA, sigma=RANDOM_PROCESS_SIGMA
)

MEMORY = TransitionBuffer(REPLAY_MEMORY_SIZE)

ACTION_CLIPPING = False
SIGNAL_CLIPPING = False

ROLLOUTS = 1000

# Architecture
ACTOR_ARCH_SPEC = GDKC(SingleHeadMLP, output_activation=torch.tanh)

CRITIC_ARCH_SPEC = GDKC(SingleHeadMergedInputMLP)
