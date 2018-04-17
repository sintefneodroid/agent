#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'

from abc import ABC, abstractmethod

import utilities as U


class Agent(ABC):
  """
  All agent should inherit from this class
  """

  def __init__(self, config=None, *args, **kwargs):
    self._step_i = 0
    self._rollout_i = 0
    self._end_training = False
    self._input_size = [0]
    self._output_size = [0]
    self._divide_by_zero_safety = 1e-10

    if config:
      self.set_config_attributes(config)

  def stop_training(self):
    self._end_training = True

  @abstractmethod
  def sample_action(self, state):
    raise NotImplementedError()

  @abstractmethod
  def optimise_wrt(self, error):
    raise NotImplementedError()

  @abstractmethod
  def evaluate(self, batch):
    raise NotImplementedError()

  @abstractmethod
  def rollout(self, init_obs, env):
    raise NotImplementedError()

  def set_config_attributes(self, config, **kwargs):
    if config:
      config_vars = U.get_upper_vars_of(config)
      for k, v in config_vars.items():
        self.__setattr__(f'_{str.lower(k)}', v)

    for k, v in kwargs.items():
      self.__setattr__(f'_{str.lower(k)}', v)
