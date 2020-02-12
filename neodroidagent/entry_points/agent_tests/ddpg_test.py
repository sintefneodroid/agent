#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

from torch.nn.functional import smooth_l1_loss

from neodroidagent.agents import DDPGAgent
from neodroidagent.common import (
    OffPolicyBatched,
    ParallelSession,
    SingleHeadConcatInputMLP,
    SingleHeadMLP,
    TransitionPointBuffer,
)
from neodroidagent.configs.test_reference.base_continous_test_config import *
from neodroidagent.entry_points.session_factory import session_factory
from neodroidagent.utilities.exploration import OrnsteinUhlenbeckProcess

__author__ = "Christian Heider Nielsen"

from warg import GDKC

"""
Description: Config for training
Author: Christian Heider Nielsen
"""

# General

CONFIG_NAME = __name__
import pathlib

CONFIG_FILE_PATH = pathlib.Path(__file__)

CONNECT_TO_RUNNING = False

# Optimiser
RENDER_ENVIRONMENT = True

ACTOR_OPTIMISER_SPEC = GDKC(torch.optim.Adam, lr=1e-4)
CRITIC_OPTIMISER_SPEC = GDKC(torch.optim.Adam, lr=1e-3)

RENDER_FREQUENCY = 10

# Architecture
ACTOR_ARCH_SPEC = GDKC(SingleHeadMLP, output_activation=torch.tanh)

CRITIC_ARCH_SPEC = GDKC(SingleHeadConcatInputMLP)

ddpg_config = globals()


def ddpg_run(
    skip_confirmation: bool = True,
    environment_type: Union[bool, str] = True,
    config=ddpg_config,
):
    session_factory(
        DDPGAgent,
        config,
        session=ParallelSession,
        skip_confirmation=skip_confirmation,
        environment=environment_type,
    )


def ddpg_test(config=ddpg_config):
    ddpg_run(environment_type="gym", config=ddpg_config)


if __name__ == "__main__":
    ddpg_test()
