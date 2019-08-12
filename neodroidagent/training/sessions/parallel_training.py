#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from collections import Iterable
from typing import Type

import draugr
from neodroidagent import PROJECT_APP_PATH, utilities as U
from neodroidagent.interfaces.agent import Agent
from neodroidagent.interfaces.specifications import TrainingSession
from neodroidagent.training.procedures import train_episodically
from neodroidagent.utilities import save_model
from draugr.stopping_key import add_early_stopping_key_combination
from neodroid.wrappers import NeodroidGymWrapper
from neodroid.environments.vector_environment import VectorEnvironment
from trolls.multiple_environments_wrapper import SubProcessEnvironments, make_gym_env

__author__ = 'cnheider'
__doc__ = ''


class parallelised_training(TrainingSession):
  def __init__(self,
               *,
               environments=None,
               default_num_train_envs=6,
               auto_reset_on_terminal_state=False,
               **kwargs):
    super().__init__(**kwargs)
    self.environments = environments
    self.default_num_train_envs = default_num_train_envs
    self.auto_reset_on_terminal = auto_reset_on_terminal_state

  def __call__(self,
               agent_type: Type[Agent],
               *,
               save=True,
               has_x_server=False,
               **kwargs):

    kwargs = draugr.NOD(**kwargs)

    if not self.environments:
      if '-v' in kwargs.environment_name:

        if self.default_num_train_envs > 0:
          self.environments = [make_gym_env(kwargs.environment_name) for _ in
                               range(self.default_num_train_envs)]
          self.environments = NeodroidGymWrapper(SubProcessEnvironments(self.environments,
                                                                        auto_reset_on_terminal=self.auto_reset_on_terminal))

      else:
        self.environments = VectorEnvironment(name=kwargs.environment_name,
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

    U.set_seeds(kwargs.seed)
    self.environments.seed(kwargs.seed)

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
                                                 **kwargs)
    except KeyboardInterrupt:
      for identifier, model in enumerate(agent.models):
        save_model(model, name=f'{agent}-{identifier}-interrupted', **kwargs)
      exit()
    finally:
      if listener:
        listener.stop()

    time_elapsed = time.time() - training_start_timestamp
    end_message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    print(f'\n{"-" * 9} {end_message} {"-" * 9}\n')

    if save and training_resume:
      if isinstance(training_resume.models, Iterable):
        for identifier, model in enumerate(training_resume.models):
          save_model(model, name=f'{agent}-{identifier}', **kwargs)
      else:
        save_model(training_resume.models,

                   name=f'{agent}-0', **kwargs)

        if 'stats' in training_resume:
          training_resume.stats.save(project_name=kwargs.project,
                                     config_name=kwargs.config_name,
                                     directory=kwargs.log_directory)

    self.environments.close()


if __name__ == '__main__':
  import neodroidagent.configs.agent_test_configs.pg_test_config as C
  from neodroidagent.agents.model_free.policy_optimisation.pg_agent import PGAgent

  env = VectorEnvironment(name=C.ENVIRONMENT_NAME,
                          connect_to_running=C.CONNECT_TO_RUNNING)
  env.seed(C.SEED)

  parallelised_training(training_procedure=train_episodically)(agent_type=PGAgent, config=C, environment=env)
