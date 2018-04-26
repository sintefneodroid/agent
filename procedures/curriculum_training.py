#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'
import time
from collections import namedtuple

import numpy as np
import torch
from tqdm import tqdm

import neodroid.wrappers.curriculum_wrapper as neo
from agents.pg_agent import PGAgent
from neodroid.models import Configuration
from utilities.environment_wrappers.action_encoding import BinaryActionEnvironment
from utilities.visualisation.term_plot import term_plot

# !/usr/bin/env python3
# coding=utf-8

tqdm.monitor_interval = 0

import configs.curriculum.curriculum_config as C

import utilities as U

torch.manual_seed(C.SEED)
neo.seed(C.SEED)

_episode_signals = U.Aggregator()
_episode_durations = U.Aggregator()
_signal_mas = U.Aggregator()
_value_estimates = U.Aggregator()
_entropy = U.Aggregator()
_sample_trajectory_lengths = U.Aggregator()

_environment = BinaryActionEnvironment(
    name=C.ENVIRONMENT, connect_to_running=C.CONNECT_TO_RUNNING
    )

_keep_stats = False
_plot_stats = False
_keep_seed_if_not_replaced = False

_agent = PGAgent(C)
# _agent = DDPGAgent(C)
# _agent = DQNAgent(C)

device = torch.device('cuda' if C.USE_CUDA else 'cpu')

_agent.build_agent(_environment, device)
_episode_i = 0
_step_i = 0
_signal_ma = 0

# _random_process = OrnsteinUhlenbeckProcess(0.5, size=_environment.action_space.shape[0])
_random_process = None


class InitStateDistribution(object):
  StateDist = namedtuple('StateDist', ('state', 'prob'))

  def __init__(self):
    self.state_tuples = []

  def add(self, state, prob):
    self.state_tuples.append(self.StateDist(state, prob))

  def sample(self):
    sds = self.StateDist(*zip(*self.state_tuples))
    return np.random.choice(sds.state, p=sds.prob)


def ma_stop(ma, solved_threshold=10):
  return ma >= solved_threshold


def get_default_configuration(environment):
  if environment:
    goal_pos_x = environment.description.configurable('ActorTransformX').observation
    goal_pos_z = environment.description.configurable('ActorTransformZ').observation
    initial_configuration = [
      Configuration('ActorTransformX', goal_pos_x),
      Configuration('ActorTransformZ', goal_pos_z),
      ]
    return initial_configuration


def get_initial_configuration(environment):
  if environment:
    goal_pos_x = environment.description.configurable('GoalTransformX').observation
    goal_pos_z = environment.description.configurable('GoalTransformZ').observation
    initial_configuration = [
      Configuration('ActorTransformX', goal_pos_x),
      Configuration('ActorTransformZ', goal_pos_z),
      ]
    return initial_configuration


def save_snapshot(**kwargs):
  _agent.save_model(C)
  for k, v in kwargs:
    U.save_statistic(v, k, C)
  U.save_statistic(_episode_signals.values, 'episode_signals', C)
  U.save_statistic(_episode_durations.values, 'episode_durations', C)
  U.save_statistic(_value_estimates.values, 'value_estimates', C)
  U.save_statistic(_entropy.values, 'entropys', C)
  U.save_statistic(_sample_trajectory_lengths.values, 'sample_trajectory_lengths', C)


def estimate_value(candidate):
  global _step_i, _episode_i, _signal_ma

  rollout_signals = 0
  rollout_seesion = range(1, C.CANDIDATE_ROLLOUTS + 1)
  rollout_seesion = tqdm(rollout_seesion, leave=True)
  for j in rollout_seesion:
    rollout_seesion.set_description(
        f'Candidate rollout #{j} of {C.CANDIDATE_ROLLOUTS} | '
        f'Est: {rollout_signals / C.CANDIDATE_ROLLOUTS}'
        )
    state_ob, _ = _environment.configure(state=candidate)

    signals, steps, *stats = _agent.rollout(state_ob, _environment)
    rollout_signals += signals

    _step_i += steps
    _episode_i += 1

    if _keep_stats:  ######### STATISTICS #########
      _episode_signals.append(signals)
      _episode_durations.append(steps)
      # _signal_ma = _episode_signals.moving_average()
      # _signal_mas.append(_signal_ma)
      _entropy.append(stats[0])

    if _episode_i % C.SAVE_MODEL_INTERVAL == 0:
      if _keep_stats:
        save_snapshot()
    ##############################

  return rollout_signals / C.CANDIDATE_ROLLOUTS


def main():
  l_star = C.random_motion_horizon
  training_start_timestamp = time.time()

  initial_configuration = get_initial_configuration(_environment)
  S_prev = _environment.generate_trajectory_from_configuration(
      initial_configuration, l_star, random_process=_random_process
      )
  train_iters = range(1, C.NUM_EPISODES + 1)
  train_iters = tqdm(train_iters)
  for iters in train_iters:
    if not _environment.is_connected:
      break

    S_i = []
    S_c = []

    fixed_point = True

    cs = tqdm(range(1, C.CANDIDATES_SIZE + 1), leave=True)
    for c in cs:
      if _plot_stats:
        term_plot(
            [i for i in range(1, _episode_i + 1)],
            _signal_mas.values,
            train_iters.write,
            offset=0,
            )
        train_iters.write('-' * 30)
        term_plot(
            [i for i in range(1, _episode_i + 1)],
            _entropy.values,
            train_iters.write,
            offset=0,
            )
        train_iters.set_description(
            f'Steps: {_step_i:9.0f} | Sig_MA: {_signal_ma:.2f} | Ent: {_entropy.moving_average():.2f}'
            )
        cs.set_description(
            f'Candidate #{c} of {C.CANDIDATES_SIZE} | '
            f'FP: {fixed_point}|L:{l_star},S_i:{len(S_i)}'
            )

      seed = U.sample(S_prev)
      S_c.extend(
          _environment.generate_trajectory_from_state(
              seed, l_star, random_process=_random_process
              )
          )

      candidate = U.sample(S_c)

      est = estimate_value(candidate)

      if C.low <= est <= C.high:
        S_i.append(candidate)
        l_star = C.random_motion_horizon
        fixed_point = False
      elif _keep_seed_if_not_replaced:
        S_i.append(seed)
    if fixed_point:
      S_i = _environment.generate_trajectory_from_configuration(
          initial_configuration, l_star, random_process=_random_process
          )
      l_star += 1

    S_prev = S_i

  time_elapsed = time.time() - training_start_timestamp
  message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
  print('\n{} {} {}\n'.format('-' * 9, message, '-' * 9))

  _agent.save_model(C)
  save_snapshot()


if __name__ == '__main__':
  main()
