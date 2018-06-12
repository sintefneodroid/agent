#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utilities.environment_wrappers.action_encoding import BinaryActionEncodingWrapper

__author__ = 'cnheider'
import glob
import os

import torch
import utilities as U
import gym

def train_agent(agent_type, config):
  device = torch.device('cuda' if config.USE_CUDA else 'cpu')
  U.set_seeds(config.SEED)

  env = gym.make(config.ENVIRONMENT_NAME)
  env.seed(config.SEED)

  agent = agent_type(config)
  agent.build(env, device)

  listener = U.add_early_stopping_key_combination(agent.stop_training)

  listener.start()
  try:
    models, stats, *_ = agent.train(env, config.ROLLOUTS, render=config.RENDER_ENVIRONMENT)
  finally:
    listener.stop()

  U.save_model(models, config, name=f'{type(agent)}')

  stats.save()

  env.close()

def train_agent2(env, agent, config):
  device = torch.device('cuda' if config.USE_CUDA else 'cpu')
  U.set_seeds(config.SEED)

  agent.build(env, device)

  listener = U.add_early_stopping_key_combination(agent.stop_training)

  listener.start()
  try:
    models, stats, *_ = agent.train(env, config.ROLLOUTS, render=config.RENDER_ENVIRONMENT)
  finally:
    listener.stop()

  U.save_model(models, config, name=f'{type(agent)}')

  stats.save()

  env.close()


def test_agent():
  '''

'''

  # _environment = neo.make(C.ENVIRONMENT_NAME, connect_to_running=C.CONNECT_TO_RUNNING)
  import configs.pg_config as C
  from agents.pg_agent import PGAgent

  '''_environment = BinaryActionEncodingWrapper(
      name=C.ENVIRONMENT_NAME, connect_to_running=C.CONNECT_TO_RUNNING
      )
  '''
  _environment = gym.make(C.ENVIRONMENT_NAME)
  _environment.seed(C.SEED)

  _list_of_files = glob.glob(str(C.MODEL_DIRECTORY) + '/*.model')
  _latest_model = max(_list_of_files, key=os.path.getctime)

  device = torch.device('cuda' if C.USE_CUDA else 'cpu')

  _agent = PGAgent(C)
  _agent.build(_environment, device)
  _agent.load(_latest_model, evaluation=True)

  _agent.infer(_environment)


def test_agent_main(agent, config):
  from configs.arguments import parse_arguments

  args = parse_arguments(f'{type(agent)}', config)

  for key, arg in args.__dict__.items():
    setattr(config, key, arg)

  U.sprint(f'\nUsing config: {config}\n', highlight=True, color='yellow')
  if not args.skip_confirmation:
    for key, arg in U.get_upper_vars_of(config).items():
      print(f'{key} = {arg}')
    input('\nPress Enter to begin... ')

  try:
    train_agent(agent, config)
  except KeyboardInterrupt:
    print('Stopping')

  torch.cuda.empty_cache()


if __name__ == '__main__':
  import configs.pg_config as C
  from agents.pg_agent import PGAgent
  env = BinaryActionEncodingWrapper(name=C.ENVIRONMENT_NAME,
                                    connect_to_running=C.CONNECT_TO_RUNNING)
  env.seed(C.SEED)

  _agent = PGAgent(C)

  train_agent2(env, agent=_agent, config=C)