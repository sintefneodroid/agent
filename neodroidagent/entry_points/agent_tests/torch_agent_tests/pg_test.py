#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

import torch
from neodroid.environments.environment import Environment
from neodroidagent.agents import PolicyGradientAgent
from neodroidagent.common import ParallelSession
from neodroidagent.configs.test_reference.base_dicrete_test_config import *

__author__ = "Christian Heider Nielsen"

from neodroidagent.entry_points.session_factory import session_factory
from warg import GDKC
from neodroidagent.common.session_factory.vertical.environment_session import (
    EnvironmentType,
)
from draugr.torch_utilities import CategoricalMLP

CONFIG_NAME = __name__

CONFIG_FILE_PATH = Path(__file__)

RENDER_ENVIRONMENT = True

OPTIMISER_SPEC = GDKC(torch.optim.Adam, lr=3e-4)
SCHEDULER_SPEC = None

POLICY_ARCH_SPEC = GDKC(constructor=CategoricalMLP, hidden_layers=128)

# RENDER_FREQUENCY = 1

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
