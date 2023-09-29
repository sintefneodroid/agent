#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

__doc__ = r"""
Description: Config for training
Author: Christian Heider Nielsen
"""

from typing import Union

import torch
from draugr.torch_utilities.optimisation.parameters.initialisation.fan_in_weight_init import (
    ortho_init,
)
from neodroid.environments.environment import Environment, EnvironmentType
from neodroidagent.agents import ProximalPolicyOptimizationAgent, TogglableValue
from neodroidagent.common import (
    ParallelSession,
)
from neodroidagent.common.architectures.actor_critic.fission import (
    CategoricalActorCriticFissionMLP,
    ActorCriticFissionMLP,
)
from neodroidagent.common.session_factory.vertical.procedures.training.off_policy_step_wise import (
    OffPolicyStepWise,
)
from neodroidagent.entry_points.session_factory import session_factory
from trolls.render_mode import RenderModeEnum
from warg import GDKC

# General

CONFIG_NAME = __name__
from pathlib import Path

CONFIG_FILE_PATH = Path(__file__)

CONNECT_TO_RUNNING = False
RENDER_MODE = RenderModeEnum.to_screen
# RENDER_ENVIRONMENT = True
# RENDER_FREQUENCY = 10


NUM_ENVS = 1
ENVIRONMENT_NAME = "Pendulum-v1"  # "InvertedPendulum-v2"
INITIAL_OBSERVATION_PERIOD = 0

NUM_BATCH_EPOCHS = 10
MINI_BATCH_SIZE = 64
BATCH_SIZE = 2048
OPTIMISER_SPEC = GDKC(torch.optim.Adam, lr=1e-3, eps=1e-5, weight_decay=1e-5)
CONTINUOUS_ARCH_SPEC: GDKC = GDKC(
    constructor=ActorCriticFissionMLP, hidden_layers=(256, 256), default_init=ortho_init
)
DISCRETE_ARCH_SPEC: GDKC = GDKC(
    constructor=CategoricalActorCriticFissionMLP,
    hidden_layers=(256, 256),
    default_init=ortho_init,
)
GRADIENT_NORM_CLIPPING = TogglableValue(True, 0.5)
ITERATIONS = 4000

ppo_config = globals()


def ppo_gym_test(config=None, **kwargs):
    """

    :param config:
    :type config:
    """
    if config is None:
        config = ppo_config
    ppo_run(environment=EnvironmentType.gym, config=config, **kwargs)


def ppo_run(
    skip_confirmation: bool = False,
    environment: Union[EnvironmentType, Environment] = EnvironmentType.zmq_pipe,
    config=None,
    **kwargs
):
    """

    :param environment:
    :type environment:
    :param skip_confirmation:
    :type skip_confirmation:

    :param config:
    :type config:
    """
    if config is None:
        config = ppo_config
    session_factory(
        ProximalPolicyOptimizationAgent,
        config,
        session=GDKC(ParallelSession, procedure=OffPolicyStepWise),
        environment=environment,
        skip_confirmation=skip_confirmation,
        **kwargs
    )


if __name__ == "__main__":
    ppo_gym_test()
