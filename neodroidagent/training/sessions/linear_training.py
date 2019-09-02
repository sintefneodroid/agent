#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from typing import Type

import gym

from draugr.stopping.stopping_key import add_early_stopping_key_combination
from draugr.torch_utilities import set_seeds

from neodroid.wrappers import NeodroidGymWrapper
from neodroidagent import PROJECT_APP_PATH
from neodroidagent.interfaces.specifications import TrainingSession
from neodroidagent.training.procedures import TorchAgent, train_episodically, VectorUnityEnvironment
from trolls.wrappers.vector_environments import VectorWrap
from warg.named_ordered_dictionary import NOD

__author__ = 'Christian Heider Nielsen'
__doc__ = ''


class linear_training(TrainingSession):

  def __init__(self,
               *,
               environments=None,
               **kwargs):
    super().__init__(**kwargs)
    self.environments = environments

  def __call__(self,
               agent_type: Type[TorchAgent],
               *,
               save_model:bool=False,
               has_x_server:bool=False,
               **kwargs):

    kwargs = NOD(**kwargs)

    if not self.environments and '-v' in kwargs.environment_name and not kwargs.connect_to_running:
        if '-v' in kwargs.environment_name:
          self.environments = VectorWrap(NeodroidGymWrapper(gym.make(kwargs.environment_name)))
        else:
          self.environments = VectorUnityEnvironment(name=kwargs.environment_name,
                                          connect_to_running=kwargs.connect_to_running)
    else:
      self.environments = VectorUnityEnvironment(name=kwargs.environment_name,
                                      connect_to_running=kwargs.connect_to_running)

    agent_class_name = agent_type.__name__
    model_directory = (PROJECT_APP_PATH.user_data / kwargs.environment_name /
                       agent_class_name / kwargs.load_time / 'models')
    config_directory = (PROJECT_APP_PATH.user_data / kwargs.environment_name /
                        agent_class_name / kwargs.load_time / 'configs')
    log_directory = (PROJECT_APP_PATH.user_log / kwargs.environment_name /
                     agent_class_name / kwargs.load_time)

    kwargs.log_directory = log_directory
    kwargs.config_directory = config_directory
    kwargs.model_directory = model_directory

    set_seeds(kwargs['SEED'])
    self.environments.seed(kwargs['SEED'])

    agent = agent_type(**kwargs)
    agent.build(self.environments)

    listener = add_early_stopping_key_combination(agent.stop_training, has_x_server=has_x_server)
    training_start_timestamp = time.time()

    training_resume = None
    if listener:
      listener.start()
    try:
      training_resume = self._training_procedure(agent,
                                                 self.environments,
                                                 render=kwargs.render_environment,
                                                 **kwargs)
    except KeyboardInterrupt:
      pass
    finally:
      if listener:
        listener.stop()

    time_elapsed = time.time() - training_start_timestamp
    end_message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    line_width = 9
    print(f'\n{"-" * line_width} {end_message} {"-" * line_width}\n')

    if save_model:
      agent.save(kwargs.model_directory, **kwargs)
    if training_resume and 'stats' in training_resume:
      training_resume.stats.save(project_name=kwargs.project,
                                 config_name=kwargs.config_name,
                                 directory=kwargs.log_directory)

    try:
      self.environments.close()
    except BrokenPipeError:
      pass

    exit()


if __name__ == '__main__':
  import neodroidagent.configs.agent_test_configs.pg_test_config as C
  from neodroidagent.agents.model_free.policy_optimisation.pg_agent import PGAgent

  env = VectorUnityEnvironment(name=C.ENVIRONMENT_NAME,
                          connect_to_running=C.CONNECT_TO_RUNNING)
  env.seed(C.SEED)

  linear_training(training_procedure=train_episodically)(agent_type=PGAgent,
                                                         config=C,
                                                         environment=env)
