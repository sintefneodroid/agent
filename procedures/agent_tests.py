#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Iterable
from functools import wraps
from itertools import count

import draugr

from configs import get_upper_case_vars_or_protected_of
from neodroid.wrappers.utility_wrappers.action_encoding_wrappers import BinaryActionEncodingWrapper

__author__ = 'cnheider'
import glob
import os

import torch
import utilities as U
import gym


def regular_train_agent_procedure(agent_type,
                                  config,
                                  environment=None):

  if not config.CONNECT_TO_RUNNING:
    if not environment:
      if '-v' in config.ENVIRONMENT_NAME:
        environment = gym.make(config.ENVIRONMENT_NAME)
      else:
        environment = BinaryActionEncodingWrapper(name=config.ENVIRONMENT_NAME,
                                                  connect_to_running=config.CONNECT_TO_RUNNING)
  else:
    environment = BinaryActionEncodingWrapper(name=config.ENVIRONMENT_NAME,
                                              connect_to_running=config.CONNECT_TO_RUNNING)


  U.set_seeds(config.SEED)
  environment.seed(config.SEED)

  agent = agent_type(config)
  agent.build(environment)

  listener = U.add_early_stopping_key_combination(agent.stop_training)

  listener.start()
  try:
    training_resume = agent.train(environment,
                                  rollouts=config.ROLLOUTS,
                                  render=config.RENDER_ENVIRONMENT)
  finally:
    listener.stop()

  identifier = count()
  if isinstance(training_resume.model, Iterable):
    for model in training_resume.model:
      U.save_model(model, config, name=f'{agent.__class__.__name__}-{identifier.__next__()}')
  else:
    U.save_model(training_resume.model, config, name=f'{agent.__class__.__name__}-{identifier.__next__()}')

    training_resume.stats.save(project_name=config.PROJECT,
                               config_name=config.CONFIG_NAME,
                               directory=config.LOG_DIRECTORY)

  environment.close()


def mp_train_agent_procedure(agent_type,
                             config,
                             environments=None,
                             test_environments=None):
  test_environments = [gym.make(config.ENVIRONMENT_NAME)]

  if not environments:
    if '-v' in config.ENVIRONMENT_NAME:
      # environment = gym.make(config.ENVIRONMENT_NAME)

      num_environments = 1

      def make_env(env_nam):
        @wraps(env_nam)
        def wrapper():
          env = gym.make(env_nam)
          return env

        return wrapper

      environments = [make_env(config.ENVIRONMENT_NAME) for _ in range(num_environments)]
      environments = U.SubprocVecEnv(environments)

    else:
      environments = BinaryActionEncodingWrapper(name=config.ENVIRONMENT_NAME,
                                                 connect_to_running=config.CONNECT_TO_RUNNING)

  U.set_seeds(config.SEED)
  environments.seed(config.SEED)

  agent = agent_type(config)
  agent.build(environments)

  listener = U.add_early_stopping_key_combination(agent.stop_training)

  listener.start()
  try:
    training_resume = agent.train(environments,
                                  test_environments,
                                  rollouts=config.ROLLOUTS,
                                  render=config.RENDER_ENVIRONMENT)
  finally:
    listener.stop()

  identifier = count()
  if isinstance(training_resume.model, Iterable):
    for model in training_resume.model:
      U.save_model(model, config, name=f'{agent.__class__.__name__}-{identifier.__next__()}')
  else:
    U.save_model(training_resume.model, config, name=f'{agent.__class__.__name__}-{identifier.__next__()}')

    training_resume.stats.save(project_name=config.PROJECT,
                               config_name=config.CONFIG_NAME,
                               directory=config.LOG_DIRECTORY)

  environments.close()


def agent_test_gym():
  '''

'''

  import configs.agent_test_configs.pg_test_config as C
  from agents.pg_agent import PGAgent

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
                    training_procedure=regular_train_agent_procedure):
  '''

'''
  from configs.arguments import parse_arguments

  args = parse_arguments(f'{type(agent)}', config)
  args_dict = args.__dict__

  if 'CONFIG' in args_dict.keys() and args_dict['CONFIG']:
    import importlib.util
    spec = importlib.util.spec_from_file_location('overloaded.config', args_dict['CONFIG'])
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
  else:
    for key, arg in args_dict.items():
      if key != 'CONFIG':
        setattr(config, key, arg)

  draugr.sprint(f'\nUsing config: {config}\n', highlight=True, color='yellow')
  if not args.skip_confirmation:
    for key, arg in get_upper_case_vars_or_protected_of(config).items():
      print(f'{key} = {arg}')
    input('\nPress Enter to begin... ')

  try:
    training_procedure(agent, config)
  except KeyboardInterrupt:
    print('Stopping')

  torch.cuda.empty_cache()


if __name__ == '__main__':
  import configs.agent_test_configs.pg_test_config as C
  from agents.pg_agent import PGAgent

  env = BinaryActionEncodingWrapper(name=C.ENVIRONMENT_NAME,
                                    connect_to_running=C.CONNECT_TO_RUNNING)
  env.seed(C.SEED)

  regular_train_agent_procedure(agent_type=PGAgent, config=C, environment=env)
