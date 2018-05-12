#!/usr/bin/env python3
# coding=utf-8
import configs
from agents.pg_agent import PGAgent

__author__ = 'cnheider'
import time
from collections import namedtuple
from types import coroutine

import numpy as np
import torch
from tqdm import tqdm

import neodroid.wrappers.curriculum_wrapper as neo
from neodroid.models import Configuration, ReactionParameters, Reaction, Displayable
from utilities.environment_wrappers.action_encoding import BinaryActionEnvironment
from utilities.visualisation.term_plot import term_plot

tqdm.monitor_interval = 0

import configs.curriculum.curriculum_config as C

import utilities as U

torch.manual_seed(C.SEED)
neo.seed(C.SEED)

stats = U.StatisticCollection(
    stats={
      'signals',
      'lengths',
      'entropies',
      'value_estimates',
      'sample_lengths'},
    measures={
      'variance',
      'mean'})

_keep_stats = False
_plot_stats = False
_keep_seed_if_not_replaced = False

# _random_process = OrnsteinUhlenbeckProcess(0.5, size=_environment.action_space.shape[0])
_random_process = None


def save_snapshot():
  _agent.save_model(C)
  stats.save(**configs.to_dict(C))




def main(config, agent, full_state_evaluation_frequency=2):
  _episode_i = 0
  _step_i = 0

  env = BinaryActionEnvironment(
      name=C.ENVIRONMENT_NAME, connect_to_running=C.CONNECT_TO_RUNNING
      )
  device = torch.device('cuda' if C.USE_CUDA else 'cpu')

  _agent.build_agent(env, device)

  l_star = C.RANDOM_MOTION_HORIZON
  training_start_timestamp = time.time()

  initial_configuration = U.get_initial_configuration(env)
  S_prev = env.generate_trajectory_from_configuration(
      initial_configuration, l_star, random_process=_random_process
      )
  train_session = range(1, C.ROLLOUTS + 1)
  train_session = tqdm(train_session, leave=False)

  for i in train_session:
    if not env.is_connected:
      break

    S_i = []
    S_c = []

    fixed_point = True

    if i % full_state_evaluation_frequency == 0:
      U.estimate_entire_state_space(env, agent, C, stats, save_snapshot=save_snapshot)

    num_candidates = tqdm(range(1, C.CANDIDATE_SET_SIZE + 1), leave=False)
    for c in num_candidates:
      if _plot_stats:
        t_range = [i for i in range(1, _episode_i + 1)]
        term_plot(t_range
                  ,
                  stats.sample_lengths.values,
                  printer=train_session.write
                  )
        term_plot(
            t_range,
            stats.entropies.values,
            printer=train_session.write
            )
        train_session.set_description(
            f'Steps: {_step_i:9.0f} | Ent: {stats.entropies.moving_average():.2f}'
            )
        num_candidates.set_description(
            f'Candidate #{c} of {C.CANDIDATE_SET_SIZE} | '
            f'FP: {fixed_point} | L: {l_star} | S_i: {len(S_i)}'
            )

      seed = U.sample(S_prev)
      S_c.extend(
          env.generate_trajectory_from_state(
              seed, l_star, random_process=_random_process
              )
          )

      candidate = U.sample(S_c)

      U.display_actor_configuration(env, candidate)

      est, _episode_i, _step_i = U.estimate_value(candidate, env, agent, C, stats,
                                                  save_snapshot=save_snapshot)

      if C.LOW <= est <= C.HIGH:
        S_i.append(candidate)
        l_star = C.RANDOM_MOTION_HORIZON
        fixed_point = False
      elif _keep_seed_if_not_replaced:
        S_i.append(seed)
    if fixed_point:
      S_i = env.generate_trajectory_from_configuration(
          initial_configuration, l_star, random_process=_random_process
          )
      l_star += 1

    S_prev = S_i

  time_elapsed = time.time() - training_start_timestamp
  message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
  print(f'\n{"-" * 9} {message} {"-" * 9}\n')

  agent.save_model(C)
  save_snapshot()


if __name__ == '__main__':
  import configs.curriculum.curriculum_config as C

  from configs.arguments import parse_arguments

  args = parse_arguments('PG Agent', C)

  for k, arg in args.__dict__.items():
    setattr(C, k, arg)

  print(f'Using config: {C}')
  if not args.skip_confirmation:
    for k, arg in U.get_upper_vars_of(C).items():
      print(f'{k} = {arg}')
    input('\nPress any key to begin... ')

  _agent = PGAgent(C)
  # _agent = DDPGAgent(C)
  # _agent = DQNAgent(C)

  try:
    main(C, _agent)
  except KeyboardInterrupt:
    print('Stopping')

  torch.cuda.empty_cache()
