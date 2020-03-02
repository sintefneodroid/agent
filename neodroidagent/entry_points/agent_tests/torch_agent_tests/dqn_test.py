#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

from neodroidagent.agents import DQNAgent
from neodroidagent.common import (
    OffPolicyEpisodic,
    ParallelSession,
    TransitionPointBuffer,
)
from neodroidagent.common.architectures import MLP
from neodroidagent.configs.test_reference.base_dicrete_test_config import *

__author__ = "Christian Heider Nielsen"

from neodroidagent.entry_points.session_factory import session_factory

from neodroidagent.utilities.exploration.exploration_specification import (
    ExplorationSpecification,
)

"""
Description: Config for training
Author: Christian Heider Nielsen
"""

CONFIG_NAME = __name__
import pathlib

CONFIG_FILE_PATH = pathlib.Path(__file__)

EXPLORATION_SPEC = ExplorationSpecification(0.95, 0.05, ITERATIONS)

INITIAL_OBSERVATION_PERIOD = 1
LEARNING_FREQUENCY = 1

dqn_config = globals()


def dqn_run(
    skip_confirmation: bool = True,
    environment_type: Union[bool, str] = True,
    config=dqn_config,
):
    session_factory(
        DQNAgent,
        config,
        session=ParallelSession(
            environment_name=ENVIRONMENT_NAME,
            procedure=OffPolicyEpisodic,
            environment=environment_type,
        ),
        skip_confirmation=skip_confirmation,
        environment=environment_type,
    )


def dqn_test(config=dqn_config):
    dqn_run(environment_type="gym", config=config)


if __name__ == "__main__":
    dqn_test()
