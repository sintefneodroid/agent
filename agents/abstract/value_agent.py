#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Any

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
    # assert type(steps_taken) is int

    if steps_taken == 0:
      return True

    sample = random.random()

    a = self._eps_start - self._eps_end

    b = math.exp(-1. * steps_taken / (self._eps_decay + self._divide_by_zero_safety))
    eps_threshold = self._eps_end + a * b

    return sample > eps_threshold

  @abstractmethod
  def _sample_model(self, state, *args, **kwargs) -> Any:
    raise NotImplementedError

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
