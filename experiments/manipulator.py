#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'
import time

import torch
# import configs.base_config as C
from tqdm import tqdm

from utilities.visualisation.term_plot import term_plot

tqdm.monitor_interval = 0

from neodroid.wrappers.gym_wrapper import NeodroidGymWrapper as neo
import utilities as U

def train_agent(config, agent):
  device = torch.device('cuda' if config.USE_CUDA else 'cpu')
  neo.seed(config.SEED)
  torch.manual_seed(config.SEED)

  env = neo(
      environment_name=config.ENVIRONMENT_NAME, connect_to_running=config.CONNECT_TO_RUNNING
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
  import configs.ddpg_config as C

  from configs.arguments import parse_arguments

  args = parse_arguments('Manipulator experiment', C)

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
