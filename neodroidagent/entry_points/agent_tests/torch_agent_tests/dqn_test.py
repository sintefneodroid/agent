#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

import torch
from warg import GDKC

from neodroid.environments.environment import Environment, EnvironmentType
from neodroidagent.agents import DeepQNetworkAgent
from neodroidagent.common import (
    OffPolicyEpisodic,
    ParallelSession,
    TransitionPointPrioritisedBuffer,
)
from neodroidagent.configs.test_reference.base_dicrete_test_config import *
from neodroidagent.entry_points.session_factory import session_factory
from neodroidagent.utilities.exploration.exploration_specification import (
    ExplorationSpecification,
)
from trolls.render_mode import RenderModeEnum

__author__ = "Christian Heider Nielsen"

__doc__ = r"""
Description: Config for training
Author: Christian Heider Nielsen
"""

CONFIG_NAME = __name__
from pathlib import Path

CONFIG_FILE_PATH = Path(__file__)

EXPLORATION_SPEC = ExplorationSpecification(0.95, 0.05, ITERATIONS)
OPTIMISER_SPEC = GDKC(torch.optim.Adam, lr=3e-4)

BATCH_SIZE = 512
INITIAL_OBSERVATION_PERIOD = BATCH_SIZE
RENDER_FREQUENCY = 1
LEARNING_FREQUENCY = 1
RENDER_ENVIRONMENT = True
RENDER_MODE = RenderModeEnum.to_screen
# RUN_TRAINING = False
MEMORY_BUFFER = TransitionPointPrioritisedBuffer(int(1e6))

dqn_config = globals()


def dqn_run(
    skip_confirmation: bool = False,
    environment: Union[EnvironmentType, Environment] = EnvironmentType.zmq_pipe,
    config=None,
    **kwargs
) -> None:
    if config is None:
        config = dqn_config
    session_factory(
        DeepQNetworkAgent,
        config,
        session=GDKC(
            ParallelSession,
            environment_name=ENVIRONMENT_NAME,
            procedure=OffPolicyEpisodic,
            environment=environment,
            **kwargs
        ),
        skip_confirmation=skip_confirmation,
        environment=environment,
        **kwargs
    )


def dqn_gym_test(config=None, **kwargs):
    if config is None:
        config = dqn_config
    dqn_run(environment=EnvironmentType.gym, config=config, **kwargs)


if __name__ == "__main__":
    dqn_gym_test()
