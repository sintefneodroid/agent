#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count
from typing import Any, Tuple

import numpy
from tqdm import tqdm

from draugr.visualisation import sprint
from draugr.writers import MockWriter
from draugr.writers.writer import Writer
from neodroid.environments.environment import Environment
from neodroid.interfaces import ActionSpace, ObservationSpace, SignalSpace
from neodroid.interfaces.unity_specifications import EnvironmentSnapshot

tqdm.monitor_interval = 0

__author__ = 'Christian Heider Nielsen'

from abc import ABC, abstractmethod


class Agent(ABC):
  '''
All agent should inherit from this class
'''

  end_training = False  # End Training flag

  # region Private

  def __init__(self,
               input_shape=None,
               output_shape=None,
               divide_by_zero_safety=1e-10,
               **kwargs):
    self._input_shape = input_shape
    self._output_shape = output_shape
    self._sample_i = 0
    self._update_i = 0

    self._divide_by_zero_safety = divide_by_zero_safety

    self.__set_protected_attr(**kwargs)

  def __repr__(self):
    return f'{self.__class__.__name__}'

  def __set_protected_attr(self, **kwargs) -> None:
    for k, v in kwargs.items():
      k_lowered = f'_{k.lstrip("_").lower()}'
      self.__setattr__(k_lowered, v)

  def __infer_io_shapes(self,
                        env: Environment,
                        print_inferred_io_shapes: bool = True) -> None:
    '''
Tries to infer input and output size from env if either _input_shape or _output_shape, is None or -1 (int)

:rtype: object
    '''

    if self._input_shape is None or self._input_shape == -1:
      self._input_shape = env.observation_space.shape

    if self._output_shape is None or self._output_shape == -1:
      self._output_shape = env.action_space.shape

    self._post_io_inference(env.observation_space,
                            env.action_space,
                            env.signal_space)

    # region print

    if print_inferred_io_shapes:
      sprint(f'input shape: {self._input_shape}\n'
             f'observation space: {env.observation_space}\n',
             color='green',
             bold=True,
             highlight=True)

      sprint(f'output shape: {self._output_shape}\n'
             f'action space: {env.action_space}\n',
             color='yellow',
             bold=True,
             highlight=True)

  # endregion

  # region Public

  def run(self,
          environment: Environment,
          render: bool = True) -> None:

    state = environment.reset().observables

    F = count(1)
    F = tqdm(F, leave=False, disable=not render)
    for frame_i in F:
      F.set_description(f'Frame {frame_i}')

      action, *_ = self.sample(state, no_random=True)
      state, signal, terminated, info = environment.react(action, render=render)

      if terminated.all():
        state = environment.reset().observables

  def rollout(self,
              initial_state: EnvironmentSnapshot,
              environment: Environment,
              *,
              train: bool = True,
              render: bool = False,
              **kwargs) -> Any:
    self._update_i += 1

    state = initial_state
    episode_signal = []
    episode_length = []

    T = count(1)
    T = tqdm(T, f'Rollout #{self._update_i}', leave=False, disable=not render)

    for t in T:
      self._sample_i += 1

      action = self.sample(state)
      next_state, signal, terminated, *_ = environment.react(action).to_gym_like_output()

      episode_signal.append(signal)

      if terminated.all():
        episode_length = t
        break

      state = next_state

    ep = numpy.array(episode_signal).sum(axis=0).mean()
    el = episode_length

    return ep, el

  def take_n_steps(self,
                   initial_state: EnvironmentSnapshot,
                   environment: Environment,
                   n: int = 100,
                   *,
                   train: bool = False,
                   render: bool = False,
                   **kwargs) -> Any:
    state = initial_state.observables

    accumulated_signal = 0

    transitions = []
    terminated = False
    T = tqdm(range(1, n + 1),
             f'Step #{self._sample_i} - {0}/{n}',
             leave=False,
             disable=not render)
    for _ in T:
      self._sample_i += 1
      action, *_ = self.sample(state)

      snapshot = environment.react(action)


      (successor_state, signal, terminated) = (snapshot.observables,
                                               snapshot.signal,
                                               snapshot.terminated)

      transitions.append((state,successor_state,signal,terminated))

      state = successor_state

      accumulated_signal += signal

      if terminated:
        snapshot = environment.reset()
        (state, signal, terminated) = (snapshot.observables,
                                       snapshot.signal,
                                       snapshot.terminated)

    return transitions, accumulated_signal, terminated, state

  def stop_training(self) -> None:
    self.end_training = True

  def build(self,
            env: Environment,
            **kwargs) -> None:
    self.__infer_io_shapes(env)
    self.__build__(env.observation_space,
                   env.action_space,
                   env.signal_space,
                   **kwargs)

  @property
  def input_shape(self):
    return self._input_shape

  @input_shape.setter
  def input_shape(self,
                  input_shape: Tuple[int]):
    self._input_shape = input_shape

  @property
  def output_shape(self):
    return self._output_shape

  @output_shape.setter
  def output_shape(self,
                   output_shape: Tuple[int]):
    self._output_shape = output_shape

  def sample(self,
             state: EnvironmentSnapshot,
             *args,
             no_random: bool = False,
             metric_writer: Writer = MockWriter(),
             **kwargs) -> Any:
    self._sample_i += 1
    return self._sample(state,
                        *args,
                        no_random=no_random,
                        metric_writer=metric_writer,
                        **kwargs)

  def update(self,
             *args,
             metric_writer: Writer = MockWriter(),
             **kwargs) -> None:
    self._update_i += 1
    return self._update(*args, metric_writer=metric_writer, **kwargs)

  # endregion

  # region Protected

  def _post_io_inference(self,
                         observation_space: ObservationSpace,
                         action_space: ActionSpace,
                         signal_space: SignalSpace) -> None:
    pass

  # endregion

  # region Abstract

  @abstractmethod
  def __build__(self,
                observation_space: ObservationSpace,
                action_space: ActionSpace,
                signal_space: SignalSpace,
                **kwargs) -> None:
    raise NotImplementedError

  @abstractmethod
  def load(self, *args, **kwargs) -> None:
    raise NotImplementedError

  @abstractmethod
  def save(self, *args, **kwargs) -> None:
    raise NotImplementedError

  @abstractmethod
  def _sample(self,
              state: EnvironmentSnapshot,
              *args,
              no_random: bool = False,
              metric_writer: Writer = MockWriter(),
              **kwargs) -> Any:
    raise NotImplementedError

  @abstractmethod
  def _update(self,
              *args,
              metric_writer: Writer = MockWriter(),
              **kwargs) -> None:
    raise NotImplementedError

  # endregion
