#!/usr/bin/env python3
# coding=utf-8
import math
import random

__author__ = 'cnheider'
import time
from collections import namedtuple
from types import coroutine

import numpy as np
import torch
from tqdm import tqdm

import neodroid.wrappers.curriculum_wrapper as neo
from agents.pg_agent import PGAgent
from neodroid.models import Configuration, ReactionParameters, Reaction, Displayable
from utilities.environment_wrappers.action_encoding import BinaryActionEnvironment
from utilities.visualisation.term_plot import term_plot

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

_keep_stats = False
_plot_stats = False
_keep_seed_if_not_replaced = False

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


@coroutine
def grid_world_sample_entire_configuration_space(environment):
  if environment:
    actor_x_conf=environment.description.configurable('ActorTransformX_')
    actor_z_conf=environment.description.configurable('ActorTransformZ_')
    x_space= actor_x_conf.configurable_space
    z_space= actor_z_conf.configurable_space
    for x in np.linspace(x_space.min_value,x_space.max_value,x_space.discrete_steps):
      for z in np.linspace(z_space.min_value,z_space.max_value,z_space.discrete_steps):
        initial_configuration = [
          Configuration('ActorTransformX_', x),
          Configuration('ActorTransformZ_', z),
          ]

        yield initial_configuration
  return


def get_initial_configuration(environment):
  if environment:
    goal_pos_x = environment.description.configurable('GoalTransformX_').configurable_value
    goal_pos_z = environment.description.configurable('GoalTransformZ_').configurable_value
    initial_configuration = [
      Configuration('ActorTransformX_', goal_pos_x),
      Configuration('ActorTransformZ_', goal_pos_z),
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


def estimate_value(candidate, env, agent):
  global _step_i, _episode_i, _signal_ma

  rollout_signals = 0
  rollout_session = range(1, C.CANDIDATE_ROLLOUTS + 1)
  rollout_session = tqdm(rollout_session, leave=False)
  for j in rollout_session:
    rollout_session.set_description(
        f'Candidate rollout #{j} of {C.CANDIDATE_ROLLOUTS} | '
        f'Est: {rollout_signals / C.CANDIDATE_ROLLOUTS}'
        )
    state_ob, _ = env.configure(state=candidate)

    signals, steps, *stats = agent.rollout(state_ob, env)
    rollout_signals += signals

    _step_i += steps
    _episode_i += 1

    if _keep_stats:
      _episode_signals.append(signals)
      _episode_durations.append(steps)
      # _signal_ma = _episode_signals.moving_average()
      # _signal_mas.append(_signal_ma)
      _entropy.append(stats[0])

    if _episode_i % C.SAVE_MODEL_INTERVAL == 0:
      if _keep_stats:
        save_snapshot()

  return rollout_signals / C.CANDIDATE_ROLLOUTS

def estimate_entire_state_space(env):
  d=[]
  vals=[]
  for configuration in grid_world_sample_entire_configuration_space(env):
    configure_params = ReactionParameters(
        terminable=False,
        episode_count=False,
        reset=True,
        configure=True
        )
    vec3 = (configuration[0].configurable_value,
            0,
            configuration[1].configurable_value)
    d.append(vec3)
    vals.append(random.random())
    displays= [Displayable('ScatterPlotDisplayer', (vals, d))]
    conf_reaction = Reaction(
        parameters=configure_params,
        configurations=configuration,
        displayables=displays
        )

    state_ob, _ = env.configure(conf_reaction)

def main(config, agent):
  env = BinaryActionEnvironment(
      name=C.ENVIRONMENT_NAME, connect_to_running=C.CONNECT_TO_RUNNING
      )
  device = torch.device('cuda' if C.USE_CUDA else 'cpu')

  _agent.build_agent(env, device)

  l_star = C.RANDOM_MOTION_HORIZON
  training_start_timestamp = time.time()

  while True:
    estimate_entire_state_space(env)

  initial_configuration = get_initial_configuration(env)
  S_prev = env.generate_trajectory_from_configuration(
      initial_configuration, l_star, random_process=_random_process
      )
  train_session = range(1, C.ROLLOUTS + 1)
  train_session = tqdm(train_session, leave=False)
  for _ in train_session:
    if not env.is_connected:
      break

    S_i = []
    S_c = []

    fixed_point = True

    cs = tqdm(range(1, C.CANDIDATES_SIZE + 1), leave=False)
    for c in cs:
      if _plot_stats:
        term_plot(
            [i for i in range(1, _episode_i + 1)],
            _signal_mas.values,
            printer=train_session.write
            )
        term_plot(
            [i for i in range(1, _episode_i + 1)],
            _entropy.values,
            printer=train_session.write
            )
        train_session.set_description(
            f'Steps: {_step_i:9.0f} | Sig_MA: {_signal_ma:.2f} | Ent: {_entropy.moving_average():.2f}'
            )
        cs.set_description(
            f'Candidate #{c} of {C.CANDIDATES_SIZE} | '
            f'FP: {fixed_point}|L:{l_star}|S_i:{len(S_i)}'
            )

      seed = U.sample(S_prev)
      S_c.extend(
          env.generate_trajectory_from_state(
              seed, l_star, random_process=_random_process
              )
          )

      candidate = U.sample(S_c)

      est = estimate_value(candidate, env, agent)

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
