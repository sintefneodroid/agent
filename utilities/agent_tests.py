#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count

from neodroid.wrappers.action_encoding_wrappers import BinaryActionEncodingWrapper

__author__ = 'cnheider'
import glob
import os

import torch
import utilities as U
import gym


def regular_train_agent_procedure(agent_type, config, environment=None):
  if not environment:
    if '-v' in config.ENVIRONMENT_NAME:
      environment = gym.make(config.ENVIRONMENT_NAME)
    else:
      environment = BinaryActionEncodingWrapper(name=config.ENVIRONMENT_NAME,
                                                connect_to_running=config.CONNECT_TO_RUNNING)

  U.set_seeds(config.SEED)
  environment.seed(config.SEED)

  agent = agent_type(config)
  device = torch.device('cuda' if config.USE_CUDA else 'cpu')
  agent.build(environment, device)

  listener = U.add_early_stopping_key_combination(agent.stop_training)

  listener.start()
  try:
    models, stats = agent.train(environment, config.ROLLOUTS, render=config.RENDER_ENVIRONMENT)
  finally:
    listener.stop()

  identifier = count()
  if isinstance(models, list) or isinstance(models, tuple):
    for model in models:
      U.save_model(model, config, name=f'{type(agent)}-{identifier.__next__()}')
  else:
    U.save_model(models, config, name=f'{type(agent)}-{identifier.__next__()}')

  stats.save()

  environment.close()


def test_agent_not_used():
  '''

'''

  import configs.agent_test_configs.test_pg_config as C
  from agents.pg_agent import PGAgent

  _environment = gym.make(C.ENVIRONMENT_NAME)
  _environment.seed(C.SEED)

  _list_of_files = glob.glob(str(C.MODEL_DIRECTORY) + '/*.model')
  _latest_model = max(_list_of_files, key=os.path.getctime)

  device = torch.device('cuda' if C.USE_CUDA else 'cpu')

  _agent = PGAgent(C)
  _agent.build(_environment, device)
  _agent.load(_latest_model, evaluation=True)

  _agent.infer(_environment)


def test_agent_main(agent, config, training_procedure=regular_train_agent_procedure):
  from configs.arguments import parse_arguments

  args = parse_arguments(f'{type(agent)}', config)

  if 'CONFIG' in args.__dict__.keys() and args.__dict__['CONFIG']:
    import importlib.util
    spec = importlib.util.spec_from_file_location('overloaded.config', args.__dict__['CONFIG'])
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
  else:
    for key, arg in args.__dict__.items():
      if key != 'CONFIG':
        setattr(config, key, arg)

  U.sprint(f'\nUsing config: {config}\n', highlight=True, color='yellow')
  if not args.skip_confirmation:
    for key, arg in U.get_upper_case_vars_or_protected_of(config).items():
      print(f'{key} = {arg}')
    input('\nPress Enter to begin... ')

  try:
    training_procedure(agent, config)
  except KeyboardInterrupt:
    print('Stopping')

  torch.cuda.empty_cache()


if __name__ == '__main__':
  import configs.agent_test_configs.test_pg_config as C
  from agents.pg_agent import PGAgent

  env = BinaryActionEncodingWrapper(name=C.ENVIRONMENT_NAME,
                                    connect_to_running=C.CONNECT_TO_RUNNING)
  env.seed(C.SEED)

  regular_train_agent_procedure(agent_type=PGAgent, config=C, environment=env)
