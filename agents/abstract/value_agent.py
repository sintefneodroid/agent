#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Any

from tqdm import tqdm

__author__ = 'cnheider'

import math
import random

import numpy as np
import torch

import utilities as U
from agents.abstract.agent import Agent


class ValueAgent(Agent):
  '''
All value iteration agents should inherit from this class
'''

  # region Public

  def __init__(self, config=None, *args, **kwargs):
    self._eps_start = 0
    self._eps_end = 0
    self._eps_decay = 0
    self._initial_observation_period = 0

    self._value_arch_parameters = None
    self._value_arch = None
    self._value_model = None

    self._naive_max_policy = False

    super().__init__(config, *args, **kwargs)

  def sample_action(self, state, **kwargs):
    self._step_i += 1
    if (
        self.epsilon_random(self._step_i)
        and self._step_i > self._initial_observation_period
    ):
      if self._verbose:
        print('Sampling from model')
      return self._sample_model(state)
    if self._verbose:
      print('Sampling from random process')
    return self.sample_random_process()

  def sample_random_process(self):
    sample = np.random.choice(self._output_size[0])
    return sample

  def epsilon_random(self, steps_taken):
    '''
:param steps_taken:
:return:
'''
    assert 0 <= self._eps_end <= self._eps_start

    if steps_taken == 0:
      return False

    sample = random.random()

    a = self._eps_start - self._eps_end

    b = math.exp(-1. * steps_taken / (self._eps_decay + self._divide_by_zero_safety))
    self._eps_threshold = self._eps_end + a * b

    if self._verbose:
      print(f'{sample} > {self._eps_threshold} = {sample > self._eps_threshold}, where a ={a} and b={b}')
    return sample > self._eps_threshold

  def save(self, C):
    U.save_model(self._value_model, C)

  def load(self, model_path, evaluation):
    print('Loading latest model: ' + model_path)
    self._value_model = self._value_arch(**self._value_arch_parameters)
    self._value_model.load_state_dict(torch.load(model_path))
    if evaluation:
      self._value_model = self._value_model.eval()
      self._value_model.train(False)
    if self._use_cuda:
      self._value_model = self._value_model.cuda()
    else:
      self._value_model = self._value_model.cpu()

  def train_episodically(
      self,
      _environment,
      *,
      rollouts=1000,
      render=False,
      render_frequency=100,
      stat_frequency=10,
      **kwargs
      ):
    '''
      :param _environment:
      :type _environment:
      :param rollouts:
      :type rollouts:
      :param render:
      :type render:
      :param render_frequency:
      :type render_frequency:
      :param stat_frequency:
      :type stat_frequency:
      :return:
      :rtype:
    '''

    stats = U.StatisticCollection(stats=('signal', 'duration', 'td_error'))

    E = range(1, rollouts)
    E = tqdm(E, leave=False)

    for episode_i in E:
      initial_state = _environment.reset()

      if episode_i % stat_frequency == 0:
        U.styled_term_plot_stats_shared_x(stats, printer=E.write)
        E.set_description(
            f'Episode: {episode_i}, '
            f'Running Signal: {stats.signal.running_value[-1]}, '
            f'Duration: {stats.duration.running_value[-1]}, '
            f'TD Error: {stats.td_error.running_value[-1]}'
            )

      if render and episode_i % render_frequency == 0:
        signal, dur, td_error, *extras = self.rollout(
            initial_state, _environment, render=render
            )
      else:
        signal, dur, td_error, *extras = self.rollout(initial_state, _environment)

      stats.append(signal, dur, td_error)

      if self._end_training:
        break

    return self._value_model, stats

  # endregion

  # region Abstract

  @abstractmethod
  def _sample_model(self, state, *args, **kwargs) -> Any:
    raise NotImplementedError

  # endregion

  # region Protected

  def _train(self, *args, **kwargs):
    return self.train_episodically(*args, **kwargs)

  # endregion
