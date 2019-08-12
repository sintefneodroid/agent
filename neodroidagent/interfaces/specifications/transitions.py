#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Sequence

import numpy
import torch
from attr import dataclass
from torch import Tensor

from warg.mixins import IndexDictTuplesMixin, IterValuesMixin

__author__ = 'cnheider'


@dataclass
class Transition(IterValuesMixin, IndexDictTuplesMixin):
  '''
    __slots__=['state','action','signal','successor_state','terminal']
  '''
  __slots__ = ['state', 'action', 'signal', 'successor_state', 'terminal']
  state: Any
  action: Any
  signal: Any
  successor_state: Any
  terminal: Any

  def __post_init__(self):
    if self.terminal:
      self.successor_state = None

  @property
  def non_terminal(self):
    if isinstance(self.terminal, (numpy.ndarray, Sequence)):
      if isinstance(self.terminal[0], Tensor):
        return [1 - t for t in self.terminal]
      return [numpy.invert(t) for t in self.terminal]
    elif isinstance(self.terminal, Tensor):
      return 1 - self.terminal
    return numpy.invert(self.terminal)

  @property
  def non_terminal_numerical(self):
    if isinstance(self.terminal, tuple):
      if isinstance(self.terminal[0], Tensor):
        return [(1 - t).type(torch.uint8) for t in self.terminal]
      return [numpy.invert(t).astype(numpy.uint8) for t in self.terminal]
    elif isinstance(self.terminal, Tensor):
      return (1 - self.terminal).type(torch.uint8)
    return numpy.invert(self.terminal).astype(numpy.uint8)

  def __len__(self):
    return len(self.state)


@dataclass
class ValuedTransition(Transition):
  '''
    __slots__=['state','action','signal','successor_state','terminal','action_prob','value_estimate']
  '''
  __slots__ = Transition.__slots__ + ['action_prob', 'value_estimate']
  state: Any
  action: Any
  signal: Any
  successor_state: Any
  terminal: Any
  action_prob: Any
  value_estimate: Any


@dataclass
class AdvantageDiscountedTransition(IterValuesMixin):
  '''
    __slots__=['state','action','signal','successor_state','terminal','action_prob','value_estimate']
  '''
  __slots__ = ['state', 'action', 'successor_state', 'terminal', 'action_prob',
               'value_estimate', 'discounted_return', 'advantage']
  state: Any
  action: Any
  successor_state: Any
  terminal: Any
  action_prob: Any
  value_estimate: Any
  discounted_return: Any
  advantage: Any
