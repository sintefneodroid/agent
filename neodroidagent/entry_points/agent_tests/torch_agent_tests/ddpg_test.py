#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

__doc__ = r"""
Description: Config for training
Author: Christian Heider Nielsen
"""

from typing import Union

from neodroid.environments.environment import Environment, EnvironmentType
from neodroidagent.agents import DeepDeterministicPolicyGradientAgent
from neodroidagent.common import ParallelSession
from neodroidagent.configs.test_reference.base_continous_test_config import *
from neodroidagent.entry_points.session_factory import session_factory

# General

CONFIG_NAME = __name__

CONFIG_FILE_PATH = Path(__file__)

CONNECT_TO_RUNNING = False
RENDER_MODE = RenderModeEnum.to_screen
NUM_ENVS = 1
ENVIRONMENT_NAME = "Pendulum-v1"
# ENVIRONMENT_NAME = "Pendulum-v1"  # "InvertedPendulum-v2"
# RENDER_FREQUENCY = 1
BATCH_SIZE = 256
# INITIAL_OBSERVATION_PERIOD = 0

ddpg_config = globals()


def ddpg_run(
    skip_confirmation: bool = False,
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
