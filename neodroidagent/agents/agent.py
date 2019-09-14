#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from itertools import count
from typing import Any, Tuple, Sequence

import numpy
from tqdm import tqdm

from draugr.visualisation import sprint
from draugr.writers import MockWriter
from draugr.writers.writer import Writer
from neodroid.environments.unity.vector_unity_environment import VectorUnityEnvironment
from neodroid.utilities.spaces import ActionSpace, ObservationSpace, SignalSpace
from neodroid.utilities.unity_specifications import EnvironmentSnapshot

tqdm.monitor_interval = 0

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''
Base class for all Neodroid Agents
'''


class Agent(ABC):
  '''
All agent should inherit from this class
'''

  # region Private

  def __init__(self,
               input_shape: Sequence = None,
               output_shape: Sequence = None,
               divide_by_zero_safety=1e-10,
               **kwargs):
    self._input_shape = input_shape
    self._output_shape = output_shape
    self._sample_i = 0
    self._update_i = 0

    self._divide_by_zero_safety = divide_by_zero_safety

    self.__set_protected_attr(**kwargs)

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}'

  def __set_protected_attr(self, **kwargs) -> None:
    for k, v in kwargs.items():
      k_lowered = f'_{k.lstrip("_").lower()}'
      self.__setattr__(k_lowered, v)

  def __infer_io_shapes(self,
                        observation_space,
                        action_space,
                        signal_space,
                        print_inferred_io_shapes: bool = True) -> None:
    '''
Tries to infer input and output size from env if either _input_shape or _output_shape, is None or -1 (int)

:rtype: object
    '''

    if self._input_shape is None or self._input_shape == -1:
      self._input_shape = observation_space.shape

    if self._output_shape is None or self._output_shape == -1:
      self._output_shape = action_space.shape

    # region print

    if print_inferred_io_shapes:
      sprint(f'input shape: {self._input_shape}\n'
             f'observation space: {observation_space}\n',
             color='green',
             bold=True,
             highlight=True)

      sprint(f'output shape: {self._output_shape}\n'
             f'action space: {action_space}\n',
             color='yellow',
             bold=True,
             highlight=True)

  # endregion

  # region Public

  def run(self,
          environment: VectorUnityEnvironment,
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
              environment: VectorUnityEnvironment,
              *,
              train: bool = True,
              render: bool = False,
              **kwargs) -> Any:
    self._update_i += 1

    state = initial_state.observables
    episode_signal = []
    episode_length = []

    T = count(1)
    T = tqdm(T, f'Rollout #', leave=False, disable=not render)

    for t in T:
      self._sample_i += 1

      action = self.sample(state)
      snapshot = environment.react(action)

      (next_state, signal, terminated) = (snapshot.observables,
                                          snapshot.signal,
                                          snapshot.terminated)

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
                   environment: VectorUnityEnvironment,
                   n: int = 100,
                   *,
                   train: bool = False,
                   render: bool = False,
                   **kwargs) -> Any:
    state = initial_state.observables

    accumulated_signal = []

    snapshot = None
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

      transitions.append((state, successor_state, signal, terminated))

      state = successor_state

      accumulated_signal += signal

      if numpy.array(terminated).all():
        snapshot = environment.reset()
        (state, signal, terminated) = (snapshot.observables,
                                       snapshot.signal,
                                       snapshot.terminated)

    return transitions, accumulated_signal, terminated, snapshot

  def build(self,
            observation_space,
            action_space,
            signal_space,
            **kwargs) -> None:
    self.__infer_io_shapes(observation_space,
                           action_space,
                           signal_space)
    self.__build__(observation_space,
                   action_space,
                   signal_space,
                   **kwargs)

  @property
  def input_shape(self) -> [int, ...]:
    return self._input_shape

  @input_shape.setter
  def input_shape(self,
                  input_shape: [int, ...]):
    self._input_shape = input_shape

  @property
  def output_shape(self) -> [int, ...]:
    return self._output_shape

  @output_shape.setter
  def output_shape(self,
                   output_shape: Tuple[int, ...]):
    self._output_shape = output_shape

  def sample(self,
             state: EnvironmentSnapshot,
             *args,
             no_random: bool = False,
             metric_writer: Writer = MockWriter(),
             **kwargs) -> Tuple[Sequence, Any]:
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
              **kwargs) -> Tuple[Sequence, Any]:
    raise NotImplementedError

  @abstractmethod
  def _update(self,
              *args,
              metric_writer: Writer = MockWriter(),
              **kwargs) -> None:
    raise NotImplementedError

  # endregion
