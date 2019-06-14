#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import draugr
from neodroid.wrappers.utility_wrappers.action_encoding_wrappers import BinaryActionEncodingWrapper

__author__ = 'cnheider'

import torch
from tqdm import tqdm

tqdm.monitor_interval = 0
from agent import utilities as U


def train_agent(config, agent):
  torch.manual_seed(config.SEED)

  env = BinaryActionEncodingWrapper(environment_name=config.ENVIRONMENT_NAME,
                                    connect_to_running=config.CONNECT_TO_RUNNING)
  env.seed(config.SEED)

  agent.build(env)

  listener = U.add_early_stopping_key_combination(agent.stop_training)

  if listener:
    listener.start()
  try:
    (trained_model,
     running_signals,
     running_lengths,
     *training_statistics) = agent.train(env,
                                         config.ROLLOUTS,
                                         render=config.RENDER_ENVIRONMENT
                                         )
  finally:
    if listener:
      listener.stop()

  draugr.save_statistic(running_signals,
                        stat_name='running_signals',
                        config_name=C.CONFIG_NAME,
                        project_name=C.PROJECT,
                        directory=C.LOG_DIRECTORY)
  draugr.save_statistic(running_lengths,
                        stat_name='running_lengths',
                        directory=C.LOG_DIRECTORY,
                        config_name=C.CONFIG_NAME,
                        project_name=C.PROJECT)
  U.save_model(trained_model, config)

  env.close()


if __name__ == '__main__':
  import experiments.rl.grid_world.grid_world_config as C

  from agent.configs import parse_arguments, get_upper_case_vars_or_protected_of

  args = parse_arguments('Curriculum grid world experiment', C)

  for key, arg in args.__dict__.items():
    setattr(C, key, arg)

  draugr.sprint(f'\nUsing config: {C}\n', highlight=True, color='yellow')
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
