#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'
import time

import torch
# import configs.base_config as C
from tqdm import tqdm

from utilities.visualisation.term_plot import term_plot

tqdm.monitor_interval = 0

# import configs.curriculum_config as C
# import configs.ddpg_config as C
# import configs.dqn_config as C
import configs.curriculum.curriculum_config as C

from agents.pg_agent import PGAgent

import neodroid.wrappers.gym_wrapper as neo
from utilities.environment_wrappers.action_encoding import BinaryActionEncodingWrapper
import utilities as U

torch.manual_seed(C.RANDOM_SEED)
neo.seed(C.RANDOM_SEED)

_keep_stats = True
_plot_stats = False

# _environment = gym.make('LunarLander-v2')
# _environment = neo.make(C.ENVIRONMENT)
_environment = BinaryActionEncodingWrapper(name=C.ENVIRONMENT, connect_to_running=C.CONNECT_TO_RUNNING)
# _environment = gym.make('CartPole-v0')
_environment.seed(C.RANDOM_SEED)

# C.ARCH_PARAMS['input_size'] = [4]
# C.ARCH_PARAMS['output_size'] = [_environment.action_space.n]

# early_stopping_condition = ma_stop
early_stopping_condition = None

_agent = PGAgent(C)
# _agent = DDPGAgent(C)
# _agent = DQNAgent(C)

_agent.build_model(_environment)

_episode_signals = U.Aggregator()
_episode_durations = U.Aggregator()
_signal_mas = U.Aggregator()
_entropy = U.Aggregator()


def ma_stop(ma, solved_threshold=10):
  return ma >= solved_threshold


def main():
  training_start_timestamp = time.time()
  step_i = 0
  signal_ma = 0
  episodes = tqdm(range(1, (C.NUM_EPISODES * C.CANDIDATES_SIZE * C.CANDIDATE_ROLLOUTS) + 1), leave=True)
  for episode_i in episodes:
    if _plot_stats:
      episodes.write('-' * 30)
      term_plot([i for i in range(1, episode_i + 1)], _signal_mas.values, episodes.write, offset=0)
      episodes.set_description(
          f'Steps: {episode_i:9.0f} | Sig_MA: {signal_ma:.2f}')
    if not _environment.is_connected:
      break

    initial_state = _environment.reset()

    signal, dur, *stats = _agent.rollout(initial_state, _environment)

    step_i += dur

    if _keep_stats:  ######### STATISTICS #########
      _episode_signals.append(signal)
      _episode_durations.append(dur)
      # _signal_ma = _episode_signals.moving_average()
      # _signal_mas.append(_signal_ma)
      _entropy.append(stats[0])

    if episode_i % C.SAVE_MODEL_INTERVAL == 0:
      if _keep_stats:
        save_snapshot()

  time_elapsed = time.time() - training_start_timestamp
  message = 'Training done, time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
  print('\n{} {} {}\n'.format('-' * 9, message, '-' * 9))

  save_snapshot()

  _environment.render(close=True)
  _environment.close()


def save_snapshot():
  _agent.save_model(C)

  U.save_statistic(_episode_signals.values, 'episode_signals', C)
  U.save_statistic(_episode_durations.values, 'episode_durations', C)
  U.save_statistic(_entropy.values, 'entropys', C)


if __name__ == '__main__':
  main()
