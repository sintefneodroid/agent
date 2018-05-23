#!/usr/bin/env python3
# coding=utf-8
from itertools import count
from warnings import warn

import torch
from tqdm import tqdm

__author__ = 'cnheider'

from abc import ABC, abstractmethod

import utilities as U


class Agent(ABC):
  '''
All agent should inherit from this class
'''

  def __init__(self, config=None, *args, **kwargs):
    self._step_i = 0
    self._rollout_i = 0
    self._end_training = False
    self._input_size = None
    self._output_size = None
    self._divide_by_zero_safety = 1e-10
    self._use_cuda = False
    self._device = torch.device(
        'cuda:0' if torch.cuda.is_available() and self._use_cuda else 'cpu'
        )

    self._verbose = False

    self.__local_defaults__()

    if config:
      self.set_config_attributes(config)

  def build_agent(self, env, device, **kwargs):
    self._infer_input_output_sizes(env)
    self._device = device

    self.__build_models__()

  @abstractmethod
  def __build_models__(self):
    raise NotImplementedError

  @abstractmethod
  def __local_defaults__(self):
    raise NotImplementedError

  def stop_training(self):
    self._end_training = True

  @abstractmethod
  def sample_action(self, state, *args, **kwargs):
    raise NotImplementedError()

  @abstractmethod
  def __sample_model__(self, state, *args, **kwargs):
    raise NotImplementedError()

  @abstractmethod
  def update_models(self, *args, **kwargs):
    raise NotImplementedError()

  @abstractmethod
  def __optimise_wrt__(self, error, *args, **kwargs):
    raise NotImplementedError()

  @abstractmethod
  def evaluate(self, batch, *args, **kwargs):
    raise NotImplementedError()

  @abstractmethod
  def rollout(self, init_obs, env, *args, **kwargs):
    raise NotImplementedError()

  @abstractmethod
  def __next__(self):
    raise NotImplementedError()

  def __iter__(self):
    return self

  def _infer_input_output_sizes(self, env, *args, **kwargs):
    '''
Tries to infer input and output size from env if either _input_size or _output_size, is None or -1 (int)

:rtype: object
'''
    if self._input_size is None or self._input_size == -1:
      self._input_size = env.observation_space.shape
    U.sprint(f'\nobservation dimensions: {self._input_size}\n',color='green',bold=True,highlight=True)

    if self._output_size is None or self._output_size == -1:
      if hasattr(env.action_space, 'num_binary_actions'):
        self._output_size = [env.action_space.num_binary_actions]
      elif len(env.action_space.shape) >= 1:
        self._output_size = env.action_space.shape
      else:
        self._output_size = [env.action_space.n]
    U.sprint(f'\naction dimensions: {self._output_size}\n',color='green',bold=True,highlight=True)

  def set_config_attributes(self, config, *args, **kwargs):
    if config:
      config_vars = U.get_upper_vars_of(config)
      self._check_for_duplicates_in_args(**config_vars)
      self._parse_set_attr(**config_vars)
    self._parse_set_attr(**kwargs)

  def _check_for_duplicates_in_args(self, *args, **kwargs):
    for k, v in kwargs.items():

      occur = 0

      if kwargs.get(k) is not None:
        occur += 1
      else:
        pass

      if k.isupper():
        k_lowered = f'_{k.lower()}'
        if kwargs.get(k_lowered) is not None:
          occur += 1
        else:
          pass
      else:
        k_lowered = f'{k.lstrip("_").upper()}'
        if kwargs.get(k_lowered) is not None:
          occur += 1
        else:
          pass

      if occur > 1:
        warn(
            f'Config contains hiding duplicates of {k} and {k_lowered}, {occur} times'
            )

  def _parse_set_attr(self, *args, **kwargs):
    for k, v in kwargs.items():
      if k.isupper():
        k_lowered = f'_{k.lower()}'
        self.__setattr__(k_lowered, v)
      else:
        self.__setattr__(k, v)

  def run(self, environment, render=True, *args, **kwargs):
    E = count(1)
    E = tqdm(E, leave=False)
    for episode_i in E:
      print('Episode {}'.format(episode_i))

      state = environment.reset()
      F = count(1)
      F = tqdm(F, leave=False)
      for frame_i in F:

        action = self.__sample_model__(state)
        state, reward, terminated, info = environment.step(action)
        if render:
          environment.render()

        if terminated:
          break
