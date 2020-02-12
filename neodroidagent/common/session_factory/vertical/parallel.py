#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Type, Union

from neodroid.environments.gym_environment import (
    NeodroidGymEnvironment,
    NeodroidVectorGymEnvironment,
)
from neodroid.environments.unity_environment import VectorUnityEnvironment

from neodroid.environments import Environment
from .procedures import OnPolicyEpisodic, Procedure
from .single_agent_environment_session import SingleAgentEnvironmentSession

from trolls import SubProcessEnvironments, make_gym_env
from warg import super_init_pass_on_kws

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
          """
__all__ = ["ParallelSession"]


@super_init_pass_on_kws
class ParallelSession(SingleAgentEnvironmentSession):
    def __init__(
        self,
        *,
        procedure: Union[Type[Procedure], Procedure] = OnPolicyEpisodic,
        environment_name: Union[str] = "",
        num_envs=6,
        auto_reset_on_terminal_state=False,
        environment: Union[bool, str, Environment] = False,
        **kwargs
    ):
        """

    @param procedure:
    @param environment_name:
    @param num_envs:
    @param auto_reset_on_terminal_state:
    @param environment:
    @param kwargs:
    """
        assert num_envs > 0
        if isinstance(environment, str) and environment == "gym":
            assert environment_name != ""
            environments = NeodroidVectorGymEnvironment(
                environment_name=environment_name,
                default_num_train_envs=num_envs,
                auto_reset_on_terminal_state=auto_reset_on_terminal_state,
            )
        elif isinstance(environment, bool):
            if not environment:
                assert environment_name != ""
            environments = VectorUnityEnvironment(
                name=environment_name,
                connect_to_running=environment,
                num_envs=num_envs,
                auto_reset_on_terminal_state=auto_reset_on_terminal_state,
            )
        else:
            assert isinstance(environment, Environment)
            environments = environment

        super().__init__(environments=environments, procedure=procedure, **kwargs)
