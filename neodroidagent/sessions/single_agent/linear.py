#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Type, Union

import gym

from neodroid.environments.environment import Environment
from neodroid.environments.gym_environment import NeodroidGymWrapper
from neodroid.environments.unity_environment.vector_unity_environment import VectorUnityEnvironment

from neodroidagent.procedures.training import OnPolicyEpisodic, Procedure
from neodroidagent.sessions.single_agent.environment_session import EnvironmentSession
from trolls.wrappers.vector_environments import VectorWrap
from warg.kw_passing import super_init_pass_on_kws

__author__ = 'Christian Heider Nielsen'
__doc__ = ''


@super_init_pass_on_kws
class LinearSession(EnvironmentSession):

  def __init__(self,
               *,
               environment: Union[str, Environment] = '',
               procedure: Union[Type[Procedure], Procedure] = OnPolicyEpisodic,
               connect_to_running=False,
               **kwargs):

    if isinstance(environment, str):
      if '-v' in environment and not connect_to_running:
        environment = VectorWrap(NeodroidGymWrapper(gym.make(environment)))
      else:
        environment = VectorUnityEnvironment(name=environment,
                                             connect_to_running=connect_to_running)

    super().__init__(environment=environment, procedure=procedure, **kwargs)


if __name__ == '__main__':
  import neodroidagent.configs.agent_test_configs.pg_test_config as C
  from neodroidagent.agents.torch_agents.model_free.on_policy import PGAgent

  env = VectorUnityEnvironment(name=C.ENVIRONMENT_NAME,
                               connect_to_running=C.CONNECT_TO_RUNNING)
  env.seed(C.SEED)

  LinearSession(procedure=OnPolicyEpisodic)(PGAgent,
                                            config=C,
                                            environment=env)
