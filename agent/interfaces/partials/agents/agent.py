#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from itertools import count
from typing import Any

from agent.exceptions.exceptions import HasNoEnvError
from neodroid.utilities import ActionSpace
from tqdm import tqdm
from warg.app_path import AppPath
from warg.arguments import check_for_duplicates_in_args, get_upper_case_vars_or_protected_of

tqdm.monitor_interval = 0

__author__ = 'cnheider'

from abc import ABC, abstractmethod


class Agent(ABC):
  '''
All agent should inherit from this class
'''

  end_training = False  # End Training flag

  # region Private

  def __init__(self,
               config=None,
               environment=None,
               **kwargs):
    self._input_shape = None
    self._output_shape = None
    self._step_i = 0
    self._update_i = 0

    self._divide_by_zero_safety = 1e-10
    self._environment = environment
    self._log_directory = AppPath('NeodroidAgent').user_data / str(int(time.time()))

    self.__defaults__()

    if config:
      self.set_config_attributes(config, **kwargs)

  def __next__(self):
    if self._environment:
      return self._step()
    else:
      raise HasNoEnvError

  def __iter__(self):
    if self._environment:
      self._last_state = None
      return self
    else:
      raise HasNoEnvError

  def __repr__(self):
    return f'{self.__class__.__name__}'

  def __parse_set_attr(self, **kwargs) -> None:
    for k, v in kwargs.items():
      if k.isupper():
        k_lowered = f'_{k.lower()}'
        self.__setattr__(k_lowered, v)
      else:
        self.__setattr__(k, v)

  # endregion

  # region Public

  def run(self, environment, render=True, episodes=0, *args, **kwargs) -> None:

    if episodes:
      E = range(episodes)
    else:
      E = count(1)
    E = tqdm(E, leave=False, disable=not render)
    for episode_i in E:
      E.set_description(f'Episode {episode_i}')

      state = environment.reset().observables

      F = count(1)
      F = tqdm(F, leave=False, disable=not render)
      for frame_i in F:
        F.set_description(f'Frame {frame_i}')

        action, *_ = self.sample_action(state, disallow_random_sample=True)
        state, signal, terminated, info = environment.step(action)
        if render:
          environment.render()

        if terminated.all():
          break

  def stop_training(self) -> None:
    self.end_training = True

  def build(self, env, **kwargs) -> None:
    self._environment = env
    self._infer_io_shapes(env)
    self._build(**kwargs)

  def set_config_attributes(self, config, **kwargs) -> None:
    if config:
      config_vars = get_upper_case_vars_or_protected_of(config)
      check_for_duplicates_in_args(**config_vars)
      self.__parse_set_attr(**config_vars)
    self.__parse_set_attr(**kwargs)

  @property
  def input_shape(self):
    return self._input_shape

  @input_shape.setter
  def input_shape(self, input_shape):
    self._input_shape = input_shape

  @property
  def output_shape(self):
    return self._output_shape

  @output_shape.setter
  def output_shape(self, output_shape):
    self._output_shape = output_shape

  # endregion

  # region Protected

  def _step(self):
    if self._environment:
      self._last_state = self._environment.react(self.sample_action(self._last_state))
      return self._last_state
    else:
      raise HasNoEnvError

  def _post_io_inference(self, env) -> None:
    pass

  def _infer_io_shapes(self, env) -> None:
    '''
Tries to infer input and output size from env if either _input_shape or _output_shape, is None or -1 (int)

:rtype: object
    '''

    if self._input_shape is None or self._input_shape == -1:
      if len(env.observation_space.shape) >= 1:
        self._input_shape = env.observation_space.shape
      else:
        self._input_shape = (env.observation_space.space.n, 1)

    if self._output_shape is None or self._output_shape == -1:
      if isinstance(env.action_space, ActionSpace):
        if env.action_space.is_discrete:
          self._output_shape = (env.action_space.num_discrete_actions, 1)
        else:
          self._output_shape = (env.action_space.n, 1)
      elif len(env.action_space.shape) >= 1:
        self._output_shape = env.action_space.shape
      else:
        self._output_shape = (env.action_space.n, 1)

    self._post_io_inference(env)

    # region print
    '''
    draugr.sprint(f'observation dimensions: {self._input_shape}\n'
                  f'observation_space: {env.observation_space}\n',
                  color='green',
                  bold=True,
                  highlight=True)

    draugr.sprint(f'action dimensions: {self._output_shape}\n'
                  f'action_space: {env.action_space}\n',
                  color='yellow',
                  bold=True,
                  highlight=True)
    '''
    # endregion

  # endregion

  # region Static

  @staticmethod
  def update_target(*, target_model, source_model, target_update_tau=3e-3):
    assert 0.0 <= target_update_tau <= 1.0
    for target_param, param in zip(target_model.parameters(), source_model.parameters()):
      target_param.data.copy_(target_update_tau * param.data
                              + (1 - target_update_tau) * target_param.data
                              )

  # endregion

  # region Abstract

  @abstractmethod
  def __defaults__(self) -> None:
    raise NotImplementedError

  @abstractmethod
  def evaluate(self, batch, *args, **kwargs) -> Any:
    raise NotImplementedError

  @abstractmethod
  def rollout(self, initial_state, environment, *, train=True, render=False, **kwargs) -> Any:
    raise NotImplementedError

  @abstractmethod
  def load(self, *args, **kwargs) -> None:
    raise NotImplementedError

  @abstractmethod
  def save(self, *args, **kwargs) -> None:
    raise NotImplementedError

  @abstractmethod
  def sample_action(self, state, *args, disallow_random_sample=False, **kwargs) -> Any:
    raise NotImplementedError

  @abstractmethod
  def update_models(self, *, stat_writer=None, **kwargs) -> None:
    raise NotImplementedError

  @abstractmethod
  def _build(self, **kwargs) -> None:
    raise NotImplementedError

  @abstractmethod
  def _optimise(self, error, *args, **kwargs) -> None:
    raise NotImplementedError

  # endregion
