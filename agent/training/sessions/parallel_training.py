#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from collections import Iterable
from typing import Type

import draugr
from agent import utilities as U
from agent.configs.base_config import LOAD_TIME
from agent.interfaces.agent import Agent
from agent.interfaces.specifications import TrainingSession
from agent.utilities import save_model
from agent.version import PROJECT_APP_PATH
from draugr.stopping_key import add_early_stopping_key_combination
from neodroid.environments.wrappers import NeodroidGymWrapper
from neodroid.environments.wrappers.vector_environment import VectorEnvironment
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

    MODEL_DIRECTORY = PROJECT_APP_PATH.user_data / kwargs.environment_name / type(agent_type).__name__ / LOAD_TIME\
                      / \
                      'models'
    CONFIG_DIRECTORY = PROJECT_APP_PATH.user_data /  kwargs.environment_name / type(agent_type).__name__ / LOAD_TIME / 'configs'
    LOG_DIRECTORY = PROJECT_APP_PATH.user_log /  kwargs.environment_name / type(agent_type).__name__/ LOAD_TIME

    kwargs.log_directory =LOG_DIRECTORY
    kwargs.config_directory =CONFIG_DIRECTORY
    kwargs.model_directory =MODEL_DIRECTORY

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
