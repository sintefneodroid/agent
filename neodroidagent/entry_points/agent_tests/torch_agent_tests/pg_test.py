#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

from neodroidagent.agents import PolicyGradientAgent
from neodroidagent.common import ParallelSession
from neodroidagent.common.architectures.distributional.categorical import CategoricalMLP
from neodroidagent.configs.test_reference.base_dicrete_test_config import *

__author__ = "Christian Heider Nielsen"

from neodroidagent.entry_points.session_factory import session_factory
from warg import GDKC

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
    environment_type: Union[bool, str] = True,
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
        environment=environment_type,
        **kwargs
    )


def pg_test(config=None, **kwargs) -> None:
    if config is None:
        config = pg_config
    pg_run(environment_type="gym", config=config, **kwargs)


if __name__ == "__main__":
    pg_test()
