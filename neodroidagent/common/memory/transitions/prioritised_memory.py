#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any

import numpy

__author__ = "Christian Heider Nielsen"
__all__ = ["PrioritisedMemory"]
__doc__ = r"""
  The main idea is that we prefer transitions that does not fit well to our current estimate of the Q
  function, because these are the transitions that we can learn most from. This reflects a simple intuition
  from our real world - if we encounter a situation that really differs from our expectation,
  we think about it over and over and change our model until it fits.
  """

import random

from neodroidagent.common import SumTree, ValuedTransitionPoint, TransitionPoint


class PrioritisedMemory:
    """

  """

    def __init__(
        self,
        capacity: int,
        per_epsilon: float = 0.01,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_beta_increment_per_sampling: float = 0.001,
        absolute_error_upper: float = 1.0,
    ):
        self._epsilon = per_epsilon
        self._alpha = per_alpha
        self._beta = per_beta
        self._beta_increment_per_sampling = per_beta_increment_per_sampling
        self._max_error = absolute_error_upper
        self._tree = SumTree(capacity)

    def _get_priority(self, error: float) -> float:
        error = numpy.abs(error) + self._epsilon

        if self._max_error:
            error = min(error, self._max_error)

        return error ** self._alpha

    def add_transition_point(self, sample: Any, error: float) -> Any:
        self._tree.push(sample, self._get_priority(error))

    def sample_transition_points(self, num: int) -> Any:
        segment = self._tree.total / num
        data = []
        leaf_indices = []
        # priorities = []

        self._beta = numpy.min([1.0, self._beta + self._beta_increment_per_sampling])

        for i in range(num):
            (leaf_index, _, _, data_) = self._tree.get(
                random.uniform(segment * i, segment * (i + 1)), normalised_rank=False
            )
            # priorities.append(priority)
            data.append(data_)
            leaf_indices.append(leaf_index)

        """
    sampling_probabilities = priorities / self._tree.total
    weights = numpy.power(self._tree._num_entries * sampling_probabilities, -self._beta)
    weights /= (weights.max() + 1e-10)  # Normalize for stability
    """

        self._last_leaf_indices = leaf_indices

        return TransitionPoint(*zip(*data))

    def update_this_batch(self, errors):
        for leaf_index, error in zip(self._last_leaf_indices, errors):
            self.update(leaf_index, error)

    def update(self, leaf_index: int, error: float) -> Any:
        self._tree.update_leaf(leaf_index, self._get_priority(error))

    def __len__(self):
        return len(self._tree)


if __name__ == "__main__":

    def stest_experience_buffer():
        capacity = 2 ** 8
        batch_size = 4

        rb = PrioritisedMemory(capacity)
        for i in range(capacity):
            a = TransitionPoint(i, i, i, i, i)
            rb.add_transition_point(a, sum(a))

        print(rb.sample_transition_points(batch_size))

        rb.update_this_batch(numpy.random.rand(capacity))

        print(rb.sample_transition_points(batch_size))

    stest_experience_buffer()
