#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/01/2020
           """

import logging
from typing import Union

from neodroid.environments.environment import Environment
from neodroidagent.agents import RandomAgent
from neodroidagent.common import ParallelSession
from neodroidagent.common.session_factory.vertical.environment_session import (
    EnvironmentType,
)
from neodroidagent.entry_points.session_factory import session_factory

# ENVIRONMENT_NAME = "CartPole-v1"
# RENDER_ENVIRONMENT = True
# RENDER_FREQUENCY = 1
# RENDER_MODE = RenderModeEnum.to_screen

random_config = globals()


def random_run(
    rollouts=None,
    skip_confirmation: bool = True,
    environment: Union[EnvironmentType, Environment] = EnvironmentType.zmq_pipe,
    config=None,
    **kwargs
) -> None:
    if config is None:
        config = random_config

    if rollouts:
        config.ROLLOUTS = rollouts

    logging.info("starting session")
    session_factory(
        RandomAgent,
        config,
        session=ParallelSession,
        skip_confirmation=skip_confirmation,
        environment=environment,
    )
    logging.info("finish session")


def random_gym_test(config=None, **kwargs) -> None:
    if config is None:
        config = random_config
    random_run(config=config)


if __name__ == "__main__":
    random_gym_test()
