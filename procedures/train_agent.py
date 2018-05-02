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


def ma_stop(ma, solved_threshold=10):
  return ma >= solved_threshold


def train_agent1(config, agent):
  torch.manual_seed(config.SEED)
  neo.seed(config.SEED)

  _keep_stats = True
  _plot_stats = False

  _environment = BinaryActionEncodingWrapper(
      name=config.ENVIRONMENT_NAME, connect_to_running=config.CONNECT_TO_RUNNING
      )
  _environment.seed(config.SEED)

  # early_stopping_condition = ma_stop
  early_stopping_condition = None

  agent.build_agent(_environment)

  _episode_signals = U.Aggregator()
  _episode_durations = U.Aggregator()
  _signal_mas = U.Aggregator()
  _entropy = U.Aggregator()

  training_start_timestamp = time.time()
  step_i = 0
  signal_ma = 0
  episodes = tqdm(range(1, config.ROLLOUTS + 1), leave=False)
  for episode_i in episodes:
    if _plot_stats:
      episodes.write('-' * 30)
      term_plot(
          [i for i in range(1, episode_i + 1)],
          _signal_mas.values,
          printer=episodes.write,
          )
      episodes.set_description(
          f'Steps: {episode_i:9.0f} | Sig_MA: {signal_ma:.2f}'
          )
    if not _environment.is_connected:
      break

    initial_state = _environment.reset()

    signal, dur, *stats = agent.rollout(initial_state, _environment)

    step_i += dur

    if _keep_stats:
      _episode_signals.append(signal)
      _episode_durations.append(dur)
      # _signal_ma = _episode_signals.moving_average()
      # _signal_mas.append(_signal_ma)
      _entropy.append(stats[0])

    if episode_i % config.SAVE_MODEL_INTERVAL == 0:
      if _keep_stats:
        save_snapshot(agent, _episode_signals, _episode_durations, _entropy)

  time_elapsed = time.time() - training_start_timestamp
  message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
  print(f'\n{"-" * 9} {message} {"-" * 9}\n')

  save_snapshot(agent, _episode_signals, _episode_durations, _entropy)

  _environment.render(close=True)
  _environment.close()


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
    _trained_model, training_statistics, *_ = agent.train(
        env, config.ROLLOUTS, render=config.RENDER_ENVIRONMENT
        )
  finally:
    listener.stop()

  U.save_model(_trained_model, config)

  env.close()


def save_snapshot(agent, _episode_signals, _episode_durations, _entropy):
  agent.save_model(C)

  U.save_statistic(_episode_signals.values, 'episode_signals', C)
  U.save_statistic(_episode_durations.values, 'episode_durations', C)
  U.save_statistic(_entropy.values, 'entropys', C)


if __name__ == '__main__':
  import argparse

  # import configs.curriculum_config as C
  # import configs.ddpg_config as C
  # import configs.dqn_config as C
  # import configs.curriculum.curriculum_config as C
  import configs.pg_config1 as C

  from configs.arguments import parse_arguments

  args = parse_arguments('PG Agent',C)

  for k, arg in args.__dict__.items():
    setattr(C, k, arg)

  print(f'Using config: {C}')
  if not args.skip_confirmation:
    for k, arg in U.get_upper_vars_of(C).items():
      print(f'{k} = {arg}')
    input('\nPress any key to begin... ')

  _agent = PGAgent(C)

  try:
    train_agent(C, _agent)
  except KeyboardInterrupt:
    print('Stopping')

  torch.cuda.empty_cache()
