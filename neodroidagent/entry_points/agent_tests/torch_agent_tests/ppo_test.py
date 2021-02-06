#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

from neodroidagent.agents import ProximalPolicyOptimizationAgent
from neodroidagent.common import (
    ActorCriticMLP,
    CategoricalActorCriticMLP,
    ParallelSession,
)
from neodroidagent.configs.test_reference.base_continous_test_config import *

__author__ = "Christian Heider Nielsen"

from neodroidagent.entry_points.session_factory import session_factory

"""
Description: Config for training
Author: Christian Heider Nielsen
"""

# General

CONFIG_NAME = __name__
from pathlib import Path

CONFIG_FILE_PATH = Path(__file__)

CONNECT_TO_RUNNING = False
RENDER_ENVIRONMENT = True
# RENDER_FREQUENCY = 10

INITIAL_OBSERVATION_PERIOD = 0

BATCH_SIZE = 256
OPTIMISER_SPEC = GDKC(torch.optim.Adam, lr=3e-4)
CONTINUOUS_ARCH_SPEC: GDKC = GDKC(constructor=ActorCriticMLP, hidden_layers=128)
DISCRETE_ARCH_SPEC: GDKC = GDKC(
    constructor=CategoricalActorCriticMLP, hidden_layers=128
)
# GRADIENT_NORM_CLIPPING = TogglableLowHigh(True, 0, 0.1)

ppo_config = globals()


def ppo_test(config=None, **kwargs):
    """

    @param config:
    @type config:
    """
    if config is None:
        config = ppo_config
    ppo_run(environment_type="gym", config=config, **kwargs)


def ppo_run(
    skip_confirmation: bool = True,
    environment_type: Union[bool, str] = True,
    config=None,
    **kwargs
):
    """

    @param skip_confirmation:
    @type skip_confirmation:
    @param environment_type:
    @type environment_type:
    @param config:
    @type config:
    """
    if config is None:
        config = ppo_config
    session_factory(
        ProximalPolicyOptimizationAgent,
        config,
        session=ParallelSession,
        environment=environment_type,
        skip_confirmation=skip_confirmation,
        **kwargs
    )


if __name__ == "__main__":
    ppo_test()
