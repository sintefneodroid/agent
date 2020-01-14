#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Type, Union

from neodroid.environments.gym_environment import NeodroidGymWrapper
from neodroid.environments.unity_environment.vector_unity_environment import (
    VectorUnityEnvironment,
)
from neodroidagent.agents.torch_agents.model_free.on_policy.pg_agent import PGAgent
from neodroid.environments.environment import Environment
from neodroidagent.common.procedures.training import OnPolicyEpisodic
from neodroidagent.common.sessions.single_agent.environment_session import (
    EnvironmentSession,
)
from neodroidagent.common.procedures.procedure_specification import Procedure

from trolls.multiple_environments_wrapper import SubProcessEnvironments, make_gym_env
from warg.kw_passing import super_init_pass_on_kws

__author__ = "Christian Heider Nielsen"
__doc__ = ""
__all__ = ["ParallelSession"]


@super_init_pass_on_kws
class ParallelSession(EnvironmentSession):
    def __init__(
        self,
        *,
        procedure: Union[Type[Procedure], Procedure] = OnPolicyEpisodic,
        environment_name: Union[str, Environment] = "",
        default_num_train_envs=6,
        auto_reset_on_terminal_state=False,
        environment_type: Union[bool, str] = False,
        **kwargs
    ):
        if isinstance(environment_type, str):
            assert environment_name != ""
            assert default_num_train_envs > 0
            environments = [
                make_gym_env(environment_name) for _ in range(default_num_train_envs)
            ]
            environments = NeodroidGymWrapper(
                SubProcessEnvironments(
                    environments, auto_reset_on_terminal=auto_reset_on_terminal_state
                ),
                environment_name=environment_name,
            )
        elif isinstance(environment_type, bool):
            environments = VectorUnityEnvironment(
                name=environment_name, connect_to_running=environment_type
            )
        else:
            environments = environment_name

        super().__init__(environments=environments, procedure=procedure, **kwargs)


if __name__ == "__main__":
    import neodroidagent.configs.agent_test_configs.pg_test_config as C

    env = VectorUnityEnvironment(
        name=C.ENVIRONMENT_NAME, connect_to_running=C.EnvironmentType
    )
    env.seed(C.SEED)

    ParallelSession()(PGAgent, config=C, environment=env)
