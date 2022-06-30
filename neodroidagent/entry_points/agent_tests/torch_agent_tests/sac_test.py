#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
Description: Test script and config for training a soft actor critic agent
Author: Christian Heider Nielsen
"""

from typing import Union

import torch
from draugr.torch_utilities import PreConcatInputMLP, ShallowStdNormalMLP
from warg import GDKC

from neodroid.environments.environment import Environment
from neodroidagent.agents import SoftActorCriticAgent
from neodroidagent.common import ParallelSession
from neodroidagent.common.session_factory.vertical.environment_session import (
    EnvironmentType,
)
from neodroidagent.common.session_factory.vertical.procedures.training.off_policy_step_wise import (
    OffPolicyStepWise,
)
from neodroidagent.configs.test_reference.base_continous_test_config import *
from neodroidagent.entry_points.session_factory import session_factory
from trolls.render_mode import RenderModeEnum

CONFIG_NAME = __name__

CONFIG_FILE_PATH = Path(__file__)

CONNECT_TO_RUNNING = False
RENDER_ENVIRONMENT = True
RUN_TRAINING = False
RENDER_FREQUENCY = 1
INITIAL_OBSERVATION_PERIOD = 1000
NUM_ENVS = 1
RENDER_MODE = RenderModeEnum.to_screen
ENVIRONMENT_NAME = "Pendulum-v1"  # "InvertedPendulum-v2"
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
