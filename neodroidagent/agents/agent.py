#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any, Sequence, Tuple

import numpy
from tqdm import tqdm

from draugr.writers import MockWriter, sprint
from draugr.writers.writer import Writer
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
               divide_by_zero_safety: float = 1e-10,
               signal_clipping=False,
               signal_clip_high=1.0,
               signal_clip_low=-1.0,
               grad_clip=False,
               grad_clip_low=-1,
               grad_clip_high=1,
               **kwargs):
    self._input_shape = input_shape
    self._output_shape = output_shape
    self._sample_i = 0
    self._update_i = 0
    self._signal_clipping = signal_clipping
    self._signal_clip_high = signal_clip_high
    self._signal_clip_low = signal_clip_low
    self._grad_clip = grad_clip
    self._grad_clip_low = grad_clip_low
    self._grad_clip_high = grad_clip_high

    self._divide_by_zero_safety = divide_by_zero_safety

    self.__set_protected_attr(**kwargs)

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}'

  def __set_protected_attr(self, **kwargs) -> None:
    for k, v in kwargs.items():
      k_lowered = f'_{k.lstrip("_").lower()}'
      self.__setattr__(k_lowered, v)

  def __infer_io_shapes(self,
                        observation_space: ObservationSpace,
                        action_space: ActionSpace,
                        signal_space: SignalSpace,
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

  def build(self,
            observation_space,
            action_space,
            signal_space,
            **kwargs) -> None:
    self.__infer_io_shapes(observation_space,
                           action_space,
                           signal_space)
    self.__build__(observation_space=observation_space,
                   action_space=action_space,
                   signal_space=signal_space,
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
             **kwargs) -> Tuple[Any, ...]:
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
    self._sample_i = 0
    return self._update(*args, metric_writer=metric_writer, **kwargs)

  def remember(self, *, signal, **kwargs):
    if self._signal_clipping:
      signal = numpy.clip(signal,
                          self._signal_clip_low,
                          self._signal_clip_high)

    self._remember(signal=signal, **kwargs)

  # endregion

  # region Abstract

  @abstractmethod
  def __build__(self,
                *,
                observation_space: ObservationSpace = None,
                action_space: ActionSpace = None,
                signal_space: SignalSpace = None,
                **kwargs) -> None:
    raise NotImplementedError

  @abstractmethod
  def load(self, *args, **kwargs) -> None:
    raise NotImplementedError

  @abstractmethod
  def save(self, *args, **kwargs) -> None:
    raise NotImplementedError

  @abstractmethod
  def _remember(self, *, signal, **kwargs):
    raise NotImplementedError

  @abstractmethod
  def _sample(self,
              state: EnvironmentSnapshot,
              *args,
              no_random: bool = False,
              metric_writer: Writer = MockWriter(),
              **kwargs) -> Tuple[Any, ...]:
    raise NotImplementedError

  @abstractmethod
  def _update(self,
              *args,
              metric_writer: Writer = MockWriter(),
              **kwargs) -> None:
    raise NotImplementedError

  # endregion
  @property
  def update_i(self):
    return self._update_i
