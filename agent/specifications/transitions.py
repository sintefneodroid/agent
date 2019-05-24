#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any

import numpy
from attr import dataclass

from warg.mixins import IterValuesMixin

__author__ = 'cnheider'




@dataclass
class Transition(IterValuesMixin):
  state: Any = None
  action: Any = None
  signal: Any = None
  successor_state: Any = None
  terminal: Any = None

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
  action_prob = None
  value_estimate = None
