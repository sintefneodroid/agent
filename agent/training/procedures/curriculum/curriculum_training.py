#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time

import torch
from neodroid.wrappers.curriculum_wrapper.curriculum_wrapper import \
  BinaryActionEncodingCurriculumEnvironment
from tqdm import tqdm

import agent.configs.curriculum.curriculum_config as C
import draugr
from agent.agents.pg_agent import PGAgent
from agent.exploration import sample
from neodroid import NeodroidWrapper
from samples.rl.curriculum.grid_world import (display_actor_configurations,
                                              estimate_entire_state_space,
                                              estimate_initial_state_expected_return,
                                              get_initial_configuration_from_goal,
                                              )
from warg.arguments import get_upper_case_vars_or_protected_of

__author__ = 'cnheider'

tqdm.monitor_interval = 0
torch.manual_seed(C.SEED)
# neo.seed(C.SEED)

'''stats = draugr.StatisticCollection(
    stats={
      'signals',
      'lengths',
      'entropies',
      'value_estimates',
      'sample_lengths'},
    measures={
      'variance',
      'mean'})
'''

_keep_stats = False
_plot_stats = False
_keep_seed_if_not_replaced = False

# _random_process = OrnsteinUhlenbeckProcess(0.5, size=_environment.action_space.shape[0])
_random_process = None


def save_snapshot():
  _agent.save(C)
  # stats.save(**configs.to_dict(C))


def main(config, agent, full_state_evaluation_frequency=20):
  _episode_i = 0
  _step_i = 0

  env = NeodroidWrapper(BinaryActionEncodingCurriculumEnvironment(name=config.ENVIRONMENT_NAME,
                                                                  connect_to_running=config.CONNECT_TO_RUNNING
                                                                  ))

  _agent.build(env)

  random_motion_length = C.RANDOM_MOTION_HORIZON
  training_start_timestamp = time.time()

  initial_configuration = get_initial_configuration_from_goal(env)
  print('Generating initial state from goal configuration')
  S_prev = env.generate_trajectory_from_configuration(initial_configuration,
                                                      random_motion_length,
                                                      random_process=_random_process
                                                      )

  train_session = range(1, config.ROLLOUTS + 1)
  train_session = tqdm(train_session, leave=False, disable=False)

  for i in train_session:
    if not env.is_connected:
      break

    S_initial = []
    S_candidate = []

    fixed_point = True

    if i % full_state_evaluation_frequency == 0:
      print('Estimating entire state space')
      estimate_entire_state_space(env,
                                  agent,
                                  C,
                                  # statistics=None,
                                  save_snapshot=save_snapshot)

    num_candidates = tqdm(range(1, C.CANDIDATE_SET_SIZE + 1), leave=False, disable=False)
    for c in num_candidates:
      if _plot_stats:
        # draugr.terminal_plot_stats_shared_x(stats, printer=train_session.write)
        train_session.set_description(f'Steps: {_step_i:9.0f}'
                                      # f' | Ent: {stats.entropies.calc_moving_average():.2f}'
                                      )
        num_candidates.set_description(f'Candidate #{c} of {C.CANDIDATE_SET_SIZE} | '
                                       f'FP: {fixed_point} | '
                                       f'L: {random_motion_length} | '
                                       f'S_i: {len(S_initial)}'
                                       )

      seed = sample(S_prev)
      S_candidate.extend(env.generate_trajectory_from_state(seed,
                                                            random_motion_length,
                                                            random_process=_random_process
                                                            )
                         )

      candidate = sample(S_candidate)

      est, _episode_i, _step_i = estimate_initial_state_expected_return(candidate,
                                                                        env,
                                                                        agent,
                                                                        C,
                                                                        save_snapshot=save_snapshot,
                                                                        # statistics=stats,
                                                                        train=True)

      if C.LOW <= est <= C.HIGH:
        S_initial.append(candidate)
        random_motion_length = C.RANDOM_MOTION_HORIZON
        fixed_point = False
      elif _keep_seed_if_not_replaced:
        S_initial.append(seed)

    display_actor_configurations(env, S_candidate)

    if fixed_point:
      print('Reached fixed point')
      print('Generating initial state from goal configuration')
      S_initial = env.generate_trajectory_from_configuration(initial_configuration,
                                                             random_motion_length,
                                                             random_process=_random_process
                                                             )
      random_motion_length += 1

    S_prev = S_initial

  time_elapsed = time.time() - training_start_timestamp
  message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
  print(f'\n{"-" * 9} {message} {"-" * 9}\n')

  agent.save(C)
  save_snapshot()


if __name__ == '__main__':
  import agent.configs.curriculum.curriculum_config as C

  from agent.configs import parse_arguments

  args = parse_arguments('PG Agent', C)

  for key, arg in args.__dict__.items():
    setattr(C, key, arg)

  draugr.sprint(f'\nUsing config: {C}\n', highlight=True, color='yellow')
  if not args.skip_confirmation:
    for key, arg in get_upper_case_vars_or_protected_of(C).items():
      print(f'{key} = {arg}')
    input('\nPress Enter to begin... ')

  _agent = PGAgent(C)
  # _agent = DDPGAgent(C)
  # _agent = DQNAgent(C)

  try:
    main(C, _agent)
  except KeyboardInterrupt:
    print('Stopping')

  torch.cuda.empty_cache()
