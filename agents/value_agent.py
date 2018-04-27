#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'

import random

import numpy as np
import torch
import math
import utilities as U
from agents.agent import Agent


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

    super().__init__(config, *args, **kwargs)

  def sample_action(self, state, **kwargs):
    if (
        self.epsilon_random(self._step_i)
        and self._step_i > self._initial_observation_period
    ):
      if self._verbose:
        print('Sampling from model')
      return self.__sample_model__(state)
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

  def __sample_model__(self, state, **kwargs):
    raise NotImplementedError

  def save_model(self, C):
    U.save_model(self._value_model, C)

  def load_model(self, model_path, evaluation):
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
