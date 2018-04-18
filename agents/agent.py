#!/usr/bin/env python3
# coding=utf-8
from itertools import count

from tqdm import tqdm

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
    self._input_size = None
    self._output_size = None
    self._divide_by_zero_safety = 1e-10
    self._use_cuda_if_available = False

    self.__defaults__()

    if config:
      self.set_config_attributes(config)

  @abstractmethod
  def __defaults__(self):
    raise NotImplementedError

  def stop_training(self):
    self._end_training = True

  @abstractmethod
  def sample_action(self, state, **kwargs):
    raise NotImplementedError()

  @abstractmethod
  def sample_model(self, state, **kwargs):
    raise NotImplementedError()

  @abstractmethod
  def optimise_wrt(self, error, **kwargs):
    raise NotImplementedError()

  @abstractmethod
  def evaluate(self, batch, **kwargs):
    raise NotImplementedError()

  @abstractmethod
  def rollout(self, init_obs, env, **kwargs):
    raise NotImplementedError()

  def infer_input_output_sizes(self, env, **kwargs):
    """
    Tries to infer input and output size from env if either _input_size or _output_size, is None or -1 (int)

    :rtype: object
    """
    if self._input_size is None or self._input_size == -1:
      self._input_size = env.observation_space.shape
    print('observation dimensions: ', self._input_size)

    if self._output_size is None or self._output_size == -1:
      if hasattr(env.action_space, 'num_binary_actions'):
        self._output_size = [env.action_space.num_binary_actions]
      elif len(env.action_space.shape) >= 1:
        self._output_size = env.action_space.shape
      else:
        self._output_size = [env.action_space.n]
    print('action dimensions: ', self._output_size)

  def set_config_attributes(self, config, **kwargs):
    if config:
      config_vars = U.get_upper_vars_of(config)
      for k, v in config_vars.items():
        self.__setattr__(f'_{str.lower(k)}', v)

    for k, v in kwargs.items():
      self.__setattr__(f'_{str.lower(k)}', v)

  def run(self, environment, render=True):
    E = count(1)
    E = tqdm(E, leave=True)
    for episode_i in E:
      print('Episode {}'.format(episode_i))

      state = environment.reset()
      F = count(1)
      F = tqdm(F, leave=True)
      for frame_i in F:

        action = self.sample_model(state)
        state, reward, terminated, info = environment.step(action)
        if render:
          environment.render()

        if terminated:
          break
