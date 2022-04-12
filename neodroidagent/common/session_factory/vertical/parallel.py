#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os import cpu_count
from typing import Type, Union

from neodroid.environments.droid_environment import VectorUnityEnvironment
from neodroid.environments.environment import Environment
from neodroid.environments.gym_environment import NeodroidVectorGymEnvironment
from warg import super_init_pass_on_kws

from .environment_session import EnvironmentType
from .procedures import OnPolicyEpisodic, Procedure
from .single_agent_environment_session import SingleAgentEnvironmentSession

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
          """
__all__ = ["ParallelSession"]


@super_init_pass_on_kws
class ParallelSession(SingleAgentEnvironmentSession):
    def __init__(
        self,
        *,
        environment: Union[Environment, EnvironmentType],
        procedure: Union[Type[Procedure], Procedure] = OnPolicyEpisodic,
        environment_name: Union[str] = "",
        num_envs=cpu_count(),
        auto_reset_on_terminal_state=False,
        **kwargs
    ):
        """

        :param procedure:
        :param environment_name:
        :param num_envs:
        :param auto_reset_on_terminal_state:
        :param environment:
        :param kwargs:"""
        assert num_envs > 0
        if isinstance(environment, EnvironmentType):
            if environment == EnvironmentType.gym:
                assert environment_name != ""
                environment_ = NeodroidVectorGymEnvironment(
                    environment_name=environment_name,
                    num_env=num_envs,
                    auto_reset_on_terminal_state=auto_reset_on_terminal_state,
                )
            elif (
                environment == EnvironmentType.zmq_pipe
                or environment == EnvironmentType.unity
            ):
                if not environment:
                    assert environment_name != ""
                environment_ = VectorUnityEnvironment(
                    name=environment_name,
                    connect_to_running=environment,
                    num_envs=num_envs,
                    auto_reset_on_terminal_state=auto_reset_on_terminal_state,
                )
            else:
                raise NotImplementedError
        else:
            assert isinstance(environment, Environment)
            environment_ = environment

        super().__init__(environment=environment_, procedure=procedure, **kwargs)
