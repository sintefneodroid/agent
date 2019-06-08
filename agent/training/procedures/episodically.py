#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroid import EnvironmentState

__author__ = 'cnheider'
__doc__ = ''
from tqdm import tqdm

import draugr
from agent.interfaces.specifications import TR


def train_episodically(C,
                       agent,
                       environment,
                       *,
                       rollouts=1000,
                       render_frequency=100,
                       stat_frequency=10,
                       disable_stdout=False
                       ) -> TR:
  '''
    :param C: Config
    :type C: module
    :param agent: The learning agent
    :type agent: Agent
    :param disable_stdout: Whether to disable stdout statements or not
    :type disable_stdout: bool
    :param environment: The environment the agent should interact with
    :type environment: NeodroidEnvironment
    :param rollouts: How many rollouts to train for
    :type rollouts: int
    :param render_frequency: How often to render environment
    :type render_frequency: int
    :param stat_frequency: How often to write statistics
    :type stat_frequency: int
    :return: A training resume containing the trained agents models and some statistics
    :rtype: TR
  '''

  E = range(1, rollouts)
  E = tqdm(E, leave=False)
  # with torchsnooper.snoop():
  with draugr.TensorBoardXWriter(str(C.LOG_DIRECTORY)) as stat_writer:
    for episode_i in E:
      initial_state = environment.reset()

      if render_frequency and episode_i % render_frequency == 0:
        render = True
      else:
        render = False

      if stat_frequency and episode_i % stat_frequency == 0:
        writer = stat_writer
      else:
        writer = None

      agent.rollout(initial_state,
                    environment,
                    render=render,
                    stat_writer=writer,
                    disable_stdout=disable_stdout)

      if agent._end_training:
        break

  return TR(agent.models, None)

def _inner_train(self,
                 env,
                 test_env,
                 *,
                 rollouts=1000,
                 render=False,
                 render_frequency=10,
                 stat_frequency=10
                 ):

  # stats = draugr.StatisticCollection(stats=('signal', 'duration'))

  E = range(1, rollouts)
  E = tqdm(E, desc='', leave=False, disable=not render)

  for episode_i in E:
    state = env.reset()
    if isinstance(state, EnvironmentState):
      state = state.observables


    if episode_i % stat_frequency == 0:
      # draugr.styled_terminal_plot_stats_shared_x(stats, printer=E.write)
      E.set_description(f'Epi: {episode_i}')

    signal, dur, *rollout_stats = self.rollout(state, env)

    if render and episode_i % render_frequency == 0:
      state = test_env.reset()
      signal, dur, *rollout_stats = self.rollout(state, test_env, render=render, train=False)

    # stats.append(signal, dur)

    if self._end_training:
      break

  return TR((self._actor, self._critic), None)