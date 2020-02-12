#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

from neodroidagent.agents import PGAgent
from neodroidagent.common import ParallelSession
from neodroidagent.common.architectures.distributional.categorical import CategoricalMLP
from neodroidagent.configs.test_reference.base_dicrete_test_config import *

__author__ = "Christian Heider Nielsen"

from neodroidagent.entry_points.session_factory import session_factory
from warg import GDKC

CONFIG_NAME = __name__

CONFIG_FILE_PATH = pathlib.Path(__file__)

RENDER_ENVIRONMENT = True

OPTIMISER_SPEC = GDKC(torch.optim.Adam, lr=1e-2)
SCHEDULER_SPEC = None

RENDER_FREQUENCY = 1

# Architecture
POLICY_ARCH_SPEC = GDKC(
    CategoricalMLP,
    hidden_layers=(64,),
    hidden_layer_activation=torch.relu,
    use_bias=True,
)

pg_config = globals()


def pg_run(
    skip_confirmation: bool = True,
    environment_type: Union[bool, str] = True,
    *,
    config=pg_config
) -> None:
    session_factory(
        PGAgent,
        config,
        session=ParallelSession,
        skip_confirmation=skip_confirmation,
        environment=environment_type,
    )


def pg_test(config=pg_config) -> None:
    pg_run(environment_type="gym", config=config)


if __name__ == "__main__":
    pg_test()
