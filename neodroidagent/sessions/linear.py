#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import inspect
import time
import types
from typing import Type, Union

import gym

from draugr.stopping.stopping_key import add_early_stopping_key_combination
from draugr.torch_utilities import set_seeds
from neodroid.environments.unity.vector_unity_environment import VectorUnityEnvironment
from neodroid.wrappers import NeodroidGymWrapper
from neodroidagent import PROJECT_APP_PATH
from neodroidagent.agents.torch_agent import TorchAgent
from neodroidagent.exceptions.exceptions import NoAgent
from neodroidagent.procedures.training import Episodic, Procedure
from neodroidagent.utilities.specifications import EnvironmentSession, Environment
from trolls.wrappers.vector_environments import VectorWrap
from warg.kw_passing import super_init_pass_on_kws
from warg.named_ordered_dictionary import NOD

__author__ = 'Christian Heider Nielsen'
__doc__ = ''


@super_init_pass_on_kws
class LinearSession(EnvironmentSession):

  def __init__(self,
               environment:Union[str,Environment],
               procedure: Union[Type[Procedure],Procedure] = Episodic,
               *,
               connect_to_running=False,
               **kwargs):

    kwargs = NOD(**kwargs)

    if isinstance(environment, str):
      if '-v' in environment and not connect_to_running:
        environment = VectorWrap(NeodroidGymWrapper(gym.make(environment)))
      else:
        environment = VectorUnityEnvironment(name=environment,
                                             connect_to_running=connect_to_running)

    super().__init__(environment, procedure, **kwargs)

  def __call__(self,
               agent: Type[TorchAgent],
               *,
               save_model: bool = False,
               has_x_server: bool = False,
               **kwargs):

    if agent is None:
      raise NoAgent

    if isinstance(agent, (types.ClassType)):
      kwargs = NOD(**kwargs)

      agent_class_name = agent.__name__
      model_directory = (PROJECT_APP_PATH.user_data / kwargs.environment_name /
                         agent_class_name / kwargs.load_time / 'models')
      config_directory = (PROJECT_APP_PATH.user_data / kwargs.environment_name /
                          agent_class_name / kwargs.load_time / 'configs')
      log_directory = (PROJECT_APP_PATH.user_log / kwargs.environment_name /
                       agent_class_name / kwargs.load_time)

      kwargs.log_directory = log_directory
      kwargs.config_directory = config_directory
      kwargs.model_directory = model_directory

      set_seeds(kwargs['seed'])
      self._environment.seed(kwargs['seed'])

      agent = agent(**kwargs)
      agent.build(self._environment.observation_space,
                  self._environment.action_space,
                  self._environment.signal_space)

    listener = add_early_stopping_key_combination(self._procedure.stop_procedure,
                                                  has_x_server=has_x_server)

    proc = self._procedure(agent, self._environment)

    training_start_timestamp = time.time()
    if listener:
      listener.start()

    try:
      training_resume = proc(render=kwargs.render_environment,
                             **kwargs)
      if training_resume and 'stats' in training_resume:
        training_resume.stats.save(project_name=kwargs.project,
                                   config_name=kwargs.config_name,
                                   directory=kwargs.log_directory)

    except KeyboardInterrupt:
      pass
    time_elapsed = time.time() - training_start_timestamp

    if listener:
      listener.stop()

    end_message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    line_width = 9
    print(f'\n{"-" * line_width} {end_message} {"-" * line_width}\n')

    if save_model:
      agent.save(kwargs.model_directory, **kwargs)

    try:
      self._environment.close()
    except BrokenPipeError:
      pass

    exit(0)


if __name__ == '__main__':
  import neodroidagent.configs.agent_test_configs.pg_test_config as C
  from neodroidagent.agents.model_free.on_policy import PGAgent

  env = VectorUnityEnvironment(name=C.ENVIRONMENT_NAME,
                               connect_to_running=C.CONNECT_TO_RUNNING)
  env.seed(C.SEED)

  LinearSession(procedure=Episodic)(PGAgent,
                                    config=C,
                                    environment=env)
