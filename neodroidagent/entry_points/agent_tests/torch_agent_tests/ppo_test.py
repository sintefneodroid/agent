#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

from torch import nn
from torch.nn.functional import mse_loss

from neodroidagent.agents import PPOAgent
from neodroidagent.common import OffPolicyBatched, ParallelSession
from neodroidagent.common.architectures.mlp_variants.actor_critic import (
    ActorCriticMLP,
    CategoricalActorCriticMLP,
)
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

BATCH_SIZE = 256

GRADIENT_NORM_CLIPPING = ClipFeature(True, 0, 0.1)

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
