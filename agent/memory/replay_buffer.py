#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = 'cnheider'

import random
from collections import deque

from agent.interfaces.specifications import TrajectoryPoint, Transition
from agent.memory import ExpandableCircularBuffer
from warg.arguments import namedtuple_args


class ReplayBuffer(object):

  def __init__(self, capacity=int(3e6)):
    self._buffer = deque(maxlen=capacity)

  def add(self, item):
    self._buffer.append(item)

  def sample(self, batch_size):
    assert batch_size <= len(self._buffer)
    return random.sample(self._buffer, batch_size)

  def __len__(self):
    return len(self._buffer)

  @namedtuple_args(Transition)
  def add_transition(self, transition):
    self.add(transition)

  def sample_transitions(self, num):
    values = self.sample(num)
    batch = Transition(*zip(*values))
    return batch


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


class TrajectoryBuffer(ExpandableCircularBuffer):

  @namedtuple_args(TrajectoryPoint)
  def add_point(self, point):
    self._add(point)

  def retrieve_trajectory(self):
    if len(self):
      batch = TrajectoryPoint(*zip(*self._memory))
      return batch
    return [None] * TrajectoryPoint._fields.__len__()
