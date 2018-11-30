#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utilities.memory.experimental import ReplayBuffer3

__author__ = 'cnheider'
'''
Description: ReplayMemory for storing transition tuples
Author: Christian Heider Nielsen
'''

import random

import numpy as np

from utilities.memory.data_structures.segment_tree import MinSegmentTree, SumSegmentTree
from utilities.memory.data_structures.sum_tree import SumTree
from .transition import Transition


class PrioritisedReplayMemory(object):  # Has cuda issues
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


class ReplayBuffer2(object):

  def __init__(self, size):
    '''Create Replay buffer.

Parameters
----------
size: int
    Max number of transitions to store in the buffer. When the buffer
    overflows the old memories are dropped.
'''
    self._storage = []
    self._maxsize = size
    self._next_idx = 0

  def __len__(self):
    return len(self._storage)

  def add(self, obs_t, action, signal, obs_tp1, terminal):
    data = (obs_t, action, signal, obs_tp1, terminal)

    if self._next_idx >= len(self._storage):
      self._storage.append(data)
    else:
      self._storage[self._next_idx] = data
    self._next_idx = (self._next_idx + 1) % self._maxsize

  def _encode_sample(self, idxes):
    obses_t, actions, rewards, obses_tp1, terminals = [], [], [], [], []
    for i in idxes:
      data = self._storage[i]
      obs_t, action, signal, obs_tp1, terminal = data
      obses_t.append(np.array(obs_t, copy=False))
      actions.append(np.array(action, copy=False))
      rewards.append(signal)
      obses_tp1.append(np.array(obs_tp1, copy=False))
      terminals.append(terminal)
    return np.array(obses_t), np.array(actions), np.array(rewards), np.array(
        obses_tp1
        ), np.array(
        terminals
        )

  def sample(self, batch_size):
    '''Sample a batch of experiences.

Parameters
----------
batch_size: int
    How many transitions to sample.

Returns
-------
obs_batch: np.array
    batch of observations
act_batch: np.array
    batch of actions executed given obs_batch
rew_batch: np.array
    rewards received as results of executing act_batch
next_obs_batch: np.array
    next set of observations seen after executing act_batch
done_mask: np.array
    done_mask[i] = 1 if executing act_batch[i] resulted in
    the end of an episode and 0 otherwise.
'''
    idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
    return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer3):

  def __init__(self, size, alpha):
    '''Create Prioritized Replay buffer.

Parameters
----------
size: int
    Max number of transitions to store in the buffer. When the buffer
    overflows the old memories are dropped.
alpha: float
    how much prioritization is used
    (0 - no prioritization, 1 - full prioritization)

See Also
--------
ReplayBuffer.__init__
'''
    super().__init__(size)
    assert alpha > 0
    self._alpha = alpha

    it_capacity = 1
    while it_capacity < size:
      it_capacity *= 2

    self._it_sum = SumSegmentTree(it_capacity)
    self._it_min = MinSegmentTree(it_capacity)
    self._max_priority = 1.0

  def add(self, *args, **kwargs):
    '''See ReplayBuffer.store_effect'''
    idx = self._next_idx
    super().add(*args, **kwargs)
    self._it_sum[idx] = self._max_priority ** self._alpha
    self._it_min[idx] = self._max_priority ** self._alpha

  def _sample_proportional(self, batch_size):
    res = []
    for _ in range(batch_size):
      # TODO(szymon): should we ensure no repeats?
      mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
      idx = self._it_sum.find_prefix_sum_idx(mass)
      res.append(idx)
    return res

  def sample(self, batch_size, beta):
    '''Sample a batch of experiences.

compared to ReplayBuffer.sample
it also returns importance weights and idxes
of sampled experiences.


Parameters
----------
batch_size: int
    How many transitions to sample.
beta: float
    To what degree to use importance weights
    (0 - no corrections, 1 - full correction)

Returns
-------
obs_batch: np.array
    batch of observations
act_batch: np.array
    batch of actions executed given obs_batch
rew_batch: np.array
    rewards received as results of executing act_batch
next_obs_batch: np.array
    next set of observations seen after executing act_batch
done_mask: np.array
    done_mask[i] = 1 if executing act_batch[i] resulted in
    the end of an episode and 0 otherwise.
weights: np.array
    Array of shape (batch_size,) and dtype np.float32
    denoting importance weight of each sampled transition
idxes: np.array
    Array of shape (batch_size,) and dtype np.int32
    idexes in buffer of sampled experiences
'''
    assert beta > 0

    idxes = self._sample_proportional(batch_size)

    weights = []
    p_min = self._it_min.min() / self._it_sum.sum()
    max_weight = (p_min * len(self._storage)) ** (-beta)

    for idx in idxes:
      p_sample = self._it_sum[idx] / self._it_sum.sum()
      weight = (p_sample * len(self._storage)) ** (-beta)
      weights.append(weight / max_weight)
    weights = np.array(weights)
    encoded_sample = self._encode_sample(idxes)
    return tuple(list(encoded_sample) + [weights, idxes])

  def update_priorities(self, idxes, priorities):
    '''Update priorities of sampled transitions.

sets priority of transition at index idxes[i] in buffer
to priorities[i].

Parameters
----------
idxes: [int]
    List of idxes of sampled transitions
priorities: [float]
    List of updated priorities corresponding to
    transitions at the sampled idxes denoted by
    variable `idxes`.
'''
    assert len(idxes) == len(priorities)
    for idx, priority in zip(idxes, priorities):
      assert priority > 0
      assert 0 <= idx < len(self._storage)
      self._it_sum[idx] = priority ** self._alpha
      self._it_min[idx] = priority ** self._alpha

      self._max_priority = max(self._max_priority, priority)
