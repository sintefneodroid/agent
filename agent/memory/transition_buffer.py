#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from agent.interfaces.specifications import Transition
from agent.memory import ExpandableCircularBuffer
from warg.arguments import namedtuple_args

__author__ = 'cnheider'


class TransitionBuffer(ExpandableCircularBuffer):

  def add_transitions(self, transitions):
    for t in transitions:
      self.add_transition(t)

  @namedtuple_args(Transition)
  def add_transition(self, transition):
    self._add(transition)

  def sample_transitions(self, num):
    '''Randomly sample transitions from memory.'''
    if len(self):
      batch = Transition(*zip(*self._sample(num)))
      return batch
    return [None] * Transition._fields.__len__()
