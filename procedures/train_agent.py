#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'
import time

import torch
# import configs.base_config as C
from tqdm import tqdm

from utilities.visualisation.term_plot import term_plot

tqdm.monitor_interval = 0

from agents.pg_agent import PGAgent

import neodroid.wrappers.gym_wrapper as neo
from utilities.environment_wrappers.action_encoding import BinaryActionEncodingWrapper
import utilities as U

def train_agent(config, agent):
  device = torch.device('cuda' if config.USE_CUDA else 'cpu')
  neo.seed(config.SEED)
  torch.manual_seed(config.SEED)

  env = BinaryActionEncodingWrapper(
      name=config.ENVIRONMENT_NAME, connect_to_running=config.CONNECT_TO_RUNNING
      )
  env.seed(config.SEED)

  agent.build_agent(env, device)

  listener = U.add_early_stopping_key_combination(agent.stop_training)

  listener.start()
  try:
    _trained_model, running_signals, running_lengths, *training_statistics = agent.train(
        env, config.ROLLOUTS, render=config.RENDER_ENVIRONMENT
        )
  finally:
    listener.stop()

  U.save_statistic(running_signals, 'running_signals', C)
  U.save_statistic(running_lengths, 'running_lengths', C)
  U.save_model(_trained_model, config)

  env.close()


if __name__ == '__main__':

  # import configs.curriculum_config as C
  # import configs.ddpg_config as C
  # import configs.dqn_config as C
  # import configs.curriculum.curriculum_config as C
  import configs.pg_config1 as C

  from configs.arguments import parse_arguments

  args = parse_arguments('PG Agent', C)

  for k, arg in args.__dict__.items():
    setattr(C, k, arg)

  print(f'Using config: {C}')
  if not args.skip_confirmation:
    for k, arg in U.get_upper_vars_of(C).items():
      print(f'{k} = {arg}')
    input('\nPress any key to begin... ')

  _agent = C.AGENT_TYPE(C)

  try:
    train_agent(C, _agent)
  except KeyboardInterrupt:
    print('Stopping')

  torch.cuda.empty_cache()
