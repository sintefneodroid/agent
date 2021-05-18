#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union
from neodroid.environments.environment import Environment
from neodroidagent.agents import DeepDeterministicPolicyGradientAgent
from neodroidagent.common import ParallelSession
from neodroidagent.entry_points.session_factory import session_factory
from neodroidagent.configs.test_reference.base_continous_test_config import *
from neodroidagent.common.session_factory.vertical.environment_session import (
    EnvironmentType,
)

__author__ = "Christian Heider Nielsen"

"""
Description: Config for training
Author: Christian Heider Nielsen
"""

# General

CONFIG_NAME = __name__
from pathlib import Path

CONFIG_FILE_PATH = Path(__file__)

CONNECT_TO_RUNNING = False

# RENDER_FREQUENCY = 1
BATCH_SIZE = 256

ddpg_config = globals()


def ddpg_run(
    skip_confirmation: bool = True,
    environment: Union[EnvironmentType, Environment] = EnvironmentType.zmq_pipe,
    config=None,
    **kwargs
):
    if config is None:
        config = ddpg_config
    session_factory(
        DeepDeterministicPolicyGradientAgent,
        config,
        session=ParallelSession,
        skip_confirmation=skip_confirmation,
        environment=environment,
        **kwargs
    )


def ddpg_gym_test(config=None, **kwargs):
    if config is None:
        config = ddpg_config
    ddpg_run(environment=EnvironmentType.gym, config=config, **kwargs)


if __name__ == "__main__":
    ddpg_gym_test()
