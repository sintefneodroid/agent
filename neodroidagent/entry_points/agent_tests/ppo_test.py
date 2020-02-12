#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

from torch import nn

from neodroidagent.agents import PPOAgent
from neodroidagent.common import OffPolicyBatched, ParallelSession
from neodroidagent.common.architectures.mlp_variants.actor_critic import ActorCriticMLP
from neodroidagent.configs.test_reference.base_continous_test_config import *

__author__ = "Christian Heider Nielsen"

from neodroidagent.entry_points.session_factory import session_factory

from warg import NOD

"""
Description: Config for training
Author: Christian Heider Nielsen
"""

# General

CONFIG_NAME = __name__
import pathlib

CONFIG_FILE_PATH = pathlib.Path(__file__)

CONNECT_TO_RUNNING = False
RENDER_ENVIRONMENT = True
RENDER_FREQUENCY = 10

INITIAL_OBSERVATION_PERIOD = 0

BATCH_SIZE = 4000

CRITIC_CRITERION = nn.MSELoss()

DISCOUNT_FACTOR = 0.99
NUM_STEPS_PER_BATCH = 4000

# Architecture
ARCH_SPEC = GDKC(
    ActorCriticMLP,
    NOD(
        **{
            "input_shape": None,  # Obtain from environment
            "hidden_layers": None,
            "hidden_layer_activation": torch.tanh,
            "output_shape": None,  # Obtain from environment
        }
    ),
)

ENTROPY_REG_COEF = 1e-3
OPTIMISER_SPEC = GDKC(constructor=torch.optim.Adam, lr=2e-4, eps=1e-4)

GRADIENT_NORM_CLIPPING = ClipFeature(True, 0, 1.0)

ppo_config = globals()


def ppo_test(config=ppo_config):
    ppo_run(environment_type="gym", config=config)


def ppo_run(
    skip_confirmation: bool = True,
    environment_type: Union[bool, str] = True,
    config=ppo_config,
):
    session_factory(
        PPOAgent,
        config,
        session=ParallelSession,
        environment=environment_type,
        skip_confirmation=skip_confirmation,
    )


if __name__ == "__main__":
    ppo_test()
