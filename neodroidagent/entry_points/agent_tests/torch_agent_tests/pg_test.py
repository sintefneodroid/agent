#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r""""""

from typing import Union

import torch
from draugr.torch_utilities import CategoricalMLP
from warg import GDKC

from neodroid.environments.environment import Environment
from neodroidagent.agents import PolicyGradientAgent
from neodroidagent.common import ParallelSession
from neodroidagent.common.session_factory.vertical.environment_session import (
    EnvironmentType,
)
from neodroidagent.configs.test_reference.base_dicrete_test_config import *
from neodroidagent.entry_points.session_factory import session_factory
from trolls.render_mode import RenderModeEnum

CONFIG_NAME = __name__

CONFIG_FILE_PATH = Path(__file__)

OPTIMISER_SPEC = GDKC(torch.optim.Adam, lr=3e-4)
SCHEDULER_SPEC = None

POLICY_ARCH_SPEC = GDKC(constructor=CategoricalMLP, hidden_layers=128)

RENDER_ENVIRONMENT = True
RENDER_FREQUENCY = 1
NUM_ENVS = 1
ENVIRONMENT_NAME = "CartPole-v0"
RENDER_MODE = RenderModeEnum.to_screen

pg_config = globals()


def pg_run(
    skip_confirmation: bool = True,
    environment: Union[EnvironmentType, Environment] = EnvironmentType.zmq_pipe,
    *,
    config=None,
    **kwargs
) -> None:
    if config is None:
        config = pg_config

    session_factory(
        PolicyGradientAgent,
        config,
        session=ParallelSession,
        skip_confirmation=skip_confirmation,
        environment=environment,
        **kwargs
    )


def pg_gym_test(config=None, **kwargs) -> None:
    if config is None:
        config = pg_config
    pg_run(environment=EnvironmentType.gym, config=config, **kwargs)


if __name__ == "__main__":
    pg_gym_test()
