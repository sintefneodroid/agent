#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Type, Union

import gym

from neodroid.environments import Environment
from neodroid.environments.gym_environment import NeodroidGymEnvironment
from neodroid.environments.unity_environment import VectorUnityEnvironment
from .procedures import Procedure, OnPolicyEpisodic
from .single_agent_environment_session import SingleAgentEnvironmentSession
from trolls import NormalisedActions, VectorWrap
from warg import super_init_pass_on_kws

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""
__all__ = ["LinearSession"]


@super_init_pass_on_kws
class LinearSession(SingleAgentEnvironmentSession):
    def __init__(
        self,
        *,
        environment_name: Union[str, Environment] = "Unnamed",
        procedure: Union[Type[Procedure], Procedure] = OnPolicyEpisodic,
        auto_reset_on_terminal_state=True,
        environment: Union[bool, str, Environment] = False,
        **kwargs
    ):
        """

    @param environment_name:
    @param procedure:
    @param environment_type:
    @param kwargs:
    """

        if isinstance(environment, str) and environment == "gym":
            assert environment_name != ""
            environments = VectorWrap(
                NeodroidGymEnvironment(
                    NormalisedActions(gym.make(environment_name)),
                    auto_reset_on_terminal_state=auto_reset_on_terminal_state,
                )
            )
        elif isinstance(environment, bool):
            environments = VectorUnityEnvironment(
                name=environment_name, connect_to_running=environment
            )
        else:
            assert isinstance(environment, Environment)
            environments = environment

        super().__init__(environments=environments, procedure=procedure, **kwargs)
