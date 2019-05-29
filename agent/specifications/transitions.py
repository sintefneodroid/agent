#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any

import numpy
from attr import dataclass

from warg.mixins import IterValuesMixin

__author__ = 'cnheider'


@dataclass
class Transition(IterValuesMixin):
  '''
    __slots__=['state','action','signal','successor_state','terminal']
  '''
  __slots__=['state','action','signal','successor_state','terminal']
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
    return numpy.invert(self.terminal)

  @property
  def non_terminal_numerical(self):
    return numpy.invert(self.terminal).astype(numpy.uint8)


@dataclass
class ValuedTransition(Transition):
  '''
    __slots__=['state','action','signal','successor_state','terminal','action_prob','value_estimate']
  '''
  __slots__=['state','action','signal','successor_state','terminal','action_prob','value_estimate']
  action_prob: Any
  value_estimate :Any
