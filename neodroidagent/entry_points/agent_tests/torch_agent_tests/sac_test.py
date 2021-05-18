#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

from neodroidagent.agents import SoftActorCriticAgent
from neodroidagent.common import ParallelSession, PreConcatInputMLP, ShallowStdNormalMLP
from neodroidagent.common.session_factory.vertical.procedures.training.off_policy_step_wise import (
    OffPolicyStepWise,
)
from neodroidagent.configs.test_reference.base_continous_test_config import *
from neodroidagent.common.session_factory.vertical.environment_session import (
    EnvironmentType,
)

__author__ = "Christian Heider Nielsen"
from neodroid.environments.environment import Environment
from neodroidagent.entry_points.session_factory import session_factory

"""
Description: Test script and config for training a soft actor critic agent
Author: Christian Heider Nielsen
"""

CONFIG_NAME = __name__
from pathlib import Path

CONFIG_FILE_PATH = Path(__file__)

CONNECT_TO_RUNNING = False
RENDER_ENVIRONMENT = True
RUN_TRAINING = False
RENDER_FREQUENCY = 1
INITIAL_OBSERVATION_PERIOD = 1000


ACTOR_OPTIMISER_SPEC: GDKC = GDKC(constructor=torch.optim.Adam, lr=3e-4, eps=1e-5)
CRITIC_OPTIMISER_SPEC: GDKC = GDKC(constructor=torch.optim.Adam, lr=3e-4, eps=1e-5)
ACTOR_ARCH_SPEC: GDKC = GDKC(ShallowStdNormalMLP, mean_head_activation=torch.nn.Tanh())
CRITIC_ARCH_SPEC: GDKC = GDKC(PreConcatInputMLP)
sac_config = globals()


def sac_gym_test(config=None, **kwargs):
    if config is None:
        config = sac_config
    sac_run(environment=EnvironmentType.gym, config=config, **kwargs)


def sac_run(
    skip_confirmation: bool = True,
    environment: Union[EnvironmentType, Environment] = EnvironmentType.zmq_pipe,
    config=None,
    **kwargs
):
    if config is None:
        config = sac_config
    session_factory(
        SoftActorCriticAgent,
        config,
        session=GDKC(
            ParallelSession,
            procedure=OffPolicyStepWise,
            environment_name=ENVIRONMENT_NAME,
            auto_reset_on_terminal_state=True,
            environment=environment,
            **kwargs
        ),
        skip_confirmation=skip_confirmation,
        **kwargs
    )


if __name__ == "__main__":
    sac_gym_test()
