#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from warnings import warn

from agent.utilities.memory import Transition, TrajectoryTrace

__author__ = 'cnheider'

import random


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
      num_entries = len(self._memory)

      if req_num > num_entries:
        warn(f'Buffer only has {num_entries}, returning {num_entries} entries of the requested {req_num}')
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


class TrajectoryTraceBuffer(ExpandableCircularBuffer):

  def add_trace(self, *args):
    super().add(TrajectoryTrace(*args))

  def retrieve_trajectory(self):
    batch = TrajectoryTrace(*zip(*super().sample()))
    return batch
