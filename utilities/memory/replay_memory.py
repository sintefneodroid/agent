#!/usr/bin/env python3
# coding=utf-8
"""
Description: ReplayMemory for storing transition tuples
Author: Christian Heider Nielsen
"""

import random

from utilities.memory.sum_tree import SumTree
from utilities.memory.transition import Transition


class CircularBuffer(object):
  """For storing transitions explored in the environment."""

  def __init__(self, capacity):
    self._capacity = capacity
    self._memory = []
    self._position = 0

  def add(self, value):
    """Saves a transition."""
    if len(self._memory) < self._capacity:
      self._memory.append(None)
    self._memory[self._position] = value
    self._position = (self._position + 1) % self._capacity

  def sample(self, num):
    """Randomly sample transitions from memory."""
    if num > len(self._memory):
      num = len(self._memory)
    batch = random.sample(self._memory, num)
    return batch

  def __len__(self):
    """Return the length of the memory list."""
    return len(self._memory)


class ReplayMemory(CircularBuffer):
  def add_transition(self, *args):
    super().add(Transition(*args))

  def sample_transitions(self, num):
    values = super().sample(num)
    batch = Transition(*zip(*values))
    return batch


class PrioritisedReplayMemory:  # Has cuda issues
  e = 0.01
  a = 0.6
  max_error = 0

  def __init__(self, capacity):
    self.tree = SumTree(capacity)

  def __getPriority__(self, error):
    if error > self.max_error:
      self.max_error = error
    return (error + self.e) ** self.a

  def add_transition(self, *args):

    p = self.__getPriority__(self.max_error)
    self.tree.add(p, Transition(*args))

  def sample_transitions(self, n):
    indices = []
    transitions = []
    segment = self.tree.total() / n

    for i in range(n):
      a = segment * i
      b = segment * (i + 1)

      s = random.uniform(a, b)
      (idx, p, transition) = self.tree.get(s)
      indices.append(idx)
      transitions.append(transition)

    batch = Transition(*zip(*transitions))
    # the * operator unpacks
    # a collection to arguments, see below
    # (S,A,R,S',T)^n -> (S^n,A^n,R^n,S'^n,T^n)
    return indices, batch

  def batch_update(self, indices, errors):
    for (idx, error) in zip(indices, errors):
      self.update(idx, error)

  def update(self, idx, error):
    p = self.__getPriority__(error)
    self.tree.update(idx, p)

  # def max_priority(self):
  #   return self.tree.max_priority()

  def __len__(self):
    return len(self.tree)
