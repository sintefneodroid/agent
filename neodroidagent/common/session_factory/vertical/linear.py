#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Type, Union

import gym
from neodroid.environments import Environment
from neodroid.environments.droid_environment import VectorUnityEnvironment
from neodroid.environments.gym_environment import NeodroidGymEnvironment
from trolls.gym_wrappers import NormalisedActions
from trolls.vector_environments import VectorWrap
from warg import super_init_pass_on_kws

from .environment_session import EnvironmentType
from .procedures import OnPolicyEpisodic, Procedure
from .single_agent_environment_session import SingleAgentEnvironmentSession

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""
__all__ = ["LinearSession"]


@super_init_pass_on_kws
class LinearSession(SingleAgentEnvironmentSession):
    def __init__(
        self,
        *,
        environment: Union[Environment, EnvironmentType],
        environment_name: Union[str, Environment] = "Unnamed",
        procedure: Union[Type[Procedure], Procedure] = OnPolicyEpisodic,
        auto_reset_on_terminal_state=True,
        **kwargs
    ):
        """

        :param environment_name:
        :param procedure:
        :param kwargs:"""
        if isinstance(environment, EnvironmentType):
            if environment == EnvironmentType.gym:
                assert environment_name != ""
                environment_ = VectorWrap(
                    NeodroidGymEnvironment(
                        NormalisedActions(gym.make(environment_name)),
                        auto_reset_on_terminal_state=auto_reset_on_terminal_state,
                    )
                )
            elif (
                environment == EnvironmentType.zmq_pipe
                or environment == EnvironmentType.unity
            ):
                environment_ = VectorUnityEnvironment(
                    name=environment_name, connect_to_running=environment
                )
            else:
                raise NotImplementedError
        else:
            assert isinstance(environment, Environment)
            environment_ = environment

        super().__init__(environment=environment_, procedure=procedure, **kwargs)
