#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

from neodroidagent.agents import SACAgent
from neodroidagent.common import ParallelSession
from neodroidagent.common.session_factory.vertical.procedures.training.off_policy_step_wise import (
    OffPolicyStepWise,
)
from neodroidagent.configs.test_reference.base_continous_test_config import *

__author__ = "Christian Heider Nielsen"

from neodroidagent.entry_points.session_factory import session_factory

"""
Description: Test script and config for training a soft actor critic agent
Author: Christian Heider Nielsen
"""

CONFIG_NAME = __name__
import pathlib

CONFIG_FILE_PATH = pathlib.Path(__file__)

CONNECT_TO_RUNNING = False
RENDER_ENVIRONMENT = True
RUN_TRAINING = True

INITIAL_OBSERVATION_PERIOD = 1000
RENDER_FREQUENCY = 10000

sac_config = globals()


def sac_test(config=sac_config):
    sac_run(environment_type="gym", config=config)


def sac_run(
    skip_confirmation: bool = True,
    environment_type: Union[bool, str] = True,
    config=sac_config,
):
    session_factory(
        SACAgent,
        config,
        session=ParallelSession(
            procedure=OffPolicyStepWise,
            environment_name=ENVIRONMENT_NAME,
            auto_reset_on_terminal_state=True,
            environment=environment_type,
        ),
        skip_confirmation=skip_confirmation,
    )


if __name__ == "__main__":
    sac_test()
