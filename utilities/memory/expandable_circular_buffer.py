#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

import random

from utilities.memory.transition import Transition


class ExpandableCircularBuffer(object):
  '''For storing transitions explored in the environment.'''

  def __init__(self, capacity=0):
    self._capacity = capacity
    self._memory = []
    self._position = 0

  def add(self, value):
    '''Saves a transition.'''
    if value is list:
      for val in value:
        self.add(val)
    else:
      if len(self._memory) < self._capacity or self._capacity == 0:
        self._memory.append(None)
      self._memory[self._position] = value
      self._position += 1
      if self._capacity != 0:
        self._position = self._position % self._capacity

  def append(self, values):
    self.add(values)

  def sample(self, req_num=None):
    '''Randomly sample transitions from memory.'''
    if req_num is None:
      return self._memory
    else:
      if req_num > len(self._memory):
        req_num = len(self._memory)
      batch = random.sample(self._memory, req_num)
      return batch

  def clear(self):
    del self._memory[:]
    self._position = 0

  def __len__(self):
    '''Return the length of the memory list.'''
    return len(self._memory)


class TransitionBuffer(ExpandableCircularBuffer):

  def append_transitions(self, transitions):
    super().add(transitions)

  def add_transition(self, *args):
    super().add(Transition(*args))

  def sample_transitions(self, num):
    values = super().sample(num)
    batch = Transition(*zip(*values))
    return batch
