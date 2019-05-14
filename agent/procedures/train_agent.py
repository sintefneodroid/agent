#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
from collections import Iterable
from itertools import count

import draugr
from trolls.multiple_environments_wrapper import SubProcessEnvironments, make_env
from warg import get_upper_case_vars_or_protected_of
from neodroid.wrappers import VectorWrap, NeodroidWrapper, BinaryActionEncodingWrapper
from warg import parse_arguments
from agent.utilities.exceptions.exceptions import NoTrainingProcedure

__author__ = 'cnheider'
import glob
import os

import torch
from agent import utilities as U
import gym


class TrainingProcedure(abc.ABC):
  def __init__(self, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    pass


class single_train_agent_procedure(TrainingProcedure):

  def __call__(self,
               agent_type,
               config,
               environment=None,
               save=False,
               has_x_server=False):

    if not config.CONNECT_TO_RUNNING:
      if not environment:
        if '-v' in config.ENVIRONMENT_NAME:
          environment = VectorWrap(NeodroidWrapper(gym.make(config.ENVIRONMENT_NAME)))
        else:
          environment = VectorWrap(BinaryActionEncodingWrapper(name=config.ENVIRONMENT_NAME,
                                                               connect_to_running=config.CONNECT_TO_RUNNING))
    else:
      environment = VectorWrap(BinaryActionEncodingWrapper(name=config.ENVIRONMENT_NAME,
                                                           connect_to_running=config.CONNECT_TO_RUNNING))

    U.set_seeds(config.SEED)
    environment.seed(config.SEED)

    agent = agent_type(config)
    agent.build(environment)

    listener = U.add_early_stopping_key_combination(agent.stop_training, has_x_server=save)

    if listener:
      listener.start()
    try:
      training_resume = agent.train(environment,
                                    test_env=environment,
                                    rollouts=config.ROLLOUTS,
                                    render=config.RENDER_ENVIRONMENT)
    finally:
      if listener:
        listener.stop()

    if save:
      identifier = count()
      if isinstance(training_resume.models, Iterable):
        for model in training_resume.models:
          U.save_model(model, config, name=f'{agent.__class__.__name__}-{identifier.__next__()}')
      else:
        U.save_model(training_resume.models, config,
                     name=f'{agent.__class__.__name__}-{identifier.__next__()}')

      if training_resume.stats:
        training_resume.stats.save(project_name=config.PROJECT,
                                   config_name=config.CONFIG_NAME,
                                   directory=config.LOG_DIRECTORY)

    environment.close()


class parallel_train_agent_procedure(TrainingProcedure):
  def __init__(self,
               *,
               environments=None,
               test_environments=None,
               default_num_train_envs=4,
               default_num_test_envs=1,
               auto_reset_on_terminal_state=False,
               **kwargs):
    super().__init__(**kwargs)
    self.environments = environments
    self.test_environments = test_environments
    self.default_num_train_envs = default_num_train_envs
    self.default_num_test_envs = default_num_test_envs
    self.auto_reset_on_terminal = auto_reset_on_terminal_state

  def __call__(self, agent_type, config, save=True, has_x_server=False):
    if not self.environments:
      if '-v' in config.ENVIRONMENT_NAME:

        if self.default_num_train_envs > 0:
          self.environments = [make_env(config.ENVIRONMENT_NAME) for _ in
                               range(self.default_num_train_envs)]
          self.environments = NeodroidWrapper(SubProcessEnvironments(self.environments,
                                                                     auto_reset_on_terminal=self.auto_reset_on_terminal))

      else:
        self.environments = BinaryActionEncodingWrapper(name=config.ENVIRONMENT_NAME,
                                                        connect_to_running=config.CONNECT_TO_RUNNING)
    if not self.test_environments:
      if '-v' in config.ENVIRONMENT_NAME:

        if self.default_num_test_envs > 0:
          self.test_environments = [make_env(config.ENVIRONMENT_NAME) for _ in
                                    range(self.default_num_test_envs)]
          self.test_environments = NeodroidWrapper(SubProcessEnvironments(self.test_environments))

      else:
        self.test_environments = BinaryActionEncodingWrapper(name=config.ENVIRONMENT_NAME,
                                                             connect_to_running=config.CONNECT_TO_RUNNING)

    U.set_seeds(config.SEED)
    self.environments.seed(config.SEED)

    agent = agent_type(config)
    agent.build(self.environments)

    listener = U.add_early_stopping_key_combination(agent.stop_training, has_x_server=has_x_server)

    if listener:
      listener.start()
    try:
      training_resume = agent.train(self.environments,
                                    self.test_environments,
                                    rollouts=config.ROLLOUTS,
                                    render=config.RENDER_ENVIRONMENT)
    except KeyboardInterrupt:
      for identifier, model in enumerate(agent.models):
        U.save_model(model, config, name=f'{agent}-{identifier}-interrupted')
      exit()
    finally:
      if listener:
        listener.stop()

    if save:
      if isinstance(training_resume.models, Iterable):
        for identifier, model in enumerate(training_resume.models):
          U.save_model(model, config, name=f'{agent}-{identifier}')
      else:
        U.save_model(training_resume.models,
                     config,
                     name=f'{agent}-0')

        if 'stats' in training_resume:
          training_resume.stats.save(project_name=config.PROJECT,
                                     config_name=config.CONFIG_NAME,
                                     directory=config.LOG_DIRECTORY)

    self.environments.close()
    self.test_environments.close()


class agent_test_gym(TrainingProcedure):
  def __call__(self, *args, **kwargs):
    '''

  '''

    import agent.configs.agent_test_configs.pg_test_config as C
    from agent.agents.pg_agent import PGAgent

    _environment = gym.make(C.ENVIRONMENT_NAME)
    _environment.seed(C.SEED)

    _list_of_files = glob.glob(str(C.MODEL_DIRECTORY) + '/*.model')
    _latest_model = max(_list_of_files, key=os.path.getctime)

    _agent = PGAgent(C)
    _agent.build(_environment)
    _agent.load(_latest_model, evaluation=True)

    _agent.infer(_environment)


def agent_test_main(agent,
                    config,
                    *,
                    training_procedure=single_train_agent_procedure,
                    parse_args=True,
                    save=True,
                    has_x_server=True,
                    skip_confirmation=False
                    ):
  '''

'''

  if training_procedure is None:
    raise NoTrainingProcedure
  elif isinstance(training_procedure, type):
    training_procedure = training_procedure()

  if parse_args:
    args = parse_arguments(f'{type(agent)}', config)
    args_dict = args.__dict__

    skip_confirmation = args.skip_confirmation

    if 'CONFIG' in args_dict.keys() and args_dict['CONFIG']:
      import importlib.util
      spec = importlib.util.spec_from_file_location('overloaded.config', args_dict['CONFIG'])
      config = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(config)
    else:
      for key, arg in args_dict.items():
        if key != 'CONFIG':
          setattr(config, key, arg)

  if has_x_server:
    display_env = os.environ['DISPLAY']
    if display_env is None:
      config.RENDER_ENVIRONMENT = False

  draugr.sprint(f'\nUsing config: {config}\n', highlight=True, color='yellow')
  if not skip_confirmation:
    for key, arg in get_upper_case_vars_or_protected_of(config).items():
      print(f'{key} = {arg}')

    print(f'\n.. Also save:{save}, has_x_server:{has_x_server}')
    input('\nPress Enter to begin... ')

  try:
    training_procedure(agent, config, save=save, has_x_server=has_x_server)
  except KeyboardInterrupt:
    print('Stopping')

  torch.cuda.empty_cache()


if __name__ == '__main__':
  import agent.configs.agent_test_configs.pg_test_config as C
  from agent.agents.pg_agent import PGAgent

  env = BinaryActionEncodingWrapper(name=C.ENVIRONMENT_NAME,
                                    connect_to_running=C.CONNECT_TO_RUNNING)
  env.seed(C.SEED)

  single_train_agent_procedure()(agent_type=PGAgent, config=C, environment=env)
