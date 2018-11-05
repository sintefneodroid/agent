#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

import torch
# import configs.base_config as C
from tqdm import tqdm

tqdm.monitor_interval = 0

import utilities as U


def train_agent(config, agent):
  device = torch.device('cuda' if config.USE_CUDA else 'cpu')
  torch.manual_seed(config.SEED)

  env = U.BinaryActionEncodingWrapper(environment_name=config.ENVIRONMENT_NAME,
                                      connect_to_running=config.CONNECT_TO_RUNNING)
  env.seed(config.SEED)

  agent.build(env, device)

  listener = U.add_early_stopping_key_combination(agent.stop_training)

  listener.start()
  try:
    (_trained_model,
     running_signals,
     running_lengths,
     *training_statistics) = agent.train(env,
                                         trials=config.ROLLOUTS,
                                         render=config.RENDER_ENVIRONMENT)
  except ValueError:
    running_signals = None
    running_lengths = None
    _trained_model = None
    print('Training procedure did not return as excepted')
  finally:
    listener.stop()

  U.save_statistic(running_signals, 'running_signals', LOG_DIRECTORY=C.LOG_DIRECTORY)
  U.save_statistic(running_lengths, 'running_lengths', LOG_DIRECTORY=C.LOG_DIRECTORY)
  U.save_model(_trained_model, config)

  env.close()


if __name__ == '__main__':
  import experiments.grid_world.grid_world_config as C

  from configs.arguments import parse_arguments, get_upper_case_vars_or_protected_of

  args = parse_arguments('Regular small grid world experiment', C)

  for key, arg in args.__dict__.items():
    setattr(C, key, arg)

  U.sprint(f'\nUsing config: {C}\n', highlight=True, color='yellow')
  if not args.skip_confirmation:
    for key, arg in get_upper_case_vars_or_protected_of(C).items():
      print(f'{key} = {arg}')
    input('\nPress Enter to begin... ')

  _agent = C.AGENT_TYPE(C)

  try:
    train_agent(C, _agent)
  except KeyboardInterrupt:
    print('Stopping')

  torch.cuda.empty_cache()
