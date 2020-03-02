#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Iterable

import numpy

__author__ = "Christian Heider Nielsen"
__all__ = ["PrioritisedBuffer"]
__doc__ = r"""

  """

import random

from neodroidagent.common.memory.data_structures.sum_tree import SumTree


class PrioritisedBuffer:
    """

"""

    def __init__(
        self,
        capacity: int,
        per_epsilon: float = 0.01,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_beta_increment_per_sampling: float = 0.000,
        max_abs_dist: float = 1.0,
    ):
        """

@param capacity:
@param per_epsilon:
@param per_alpha:
@param per_beta:
@param per_beta_increment_per_sampling:
@param max_abs_dist:
"""
        self._epsilon = per_epsilon
        self._alpha = per_alpha
        self._beta = per_beta
        self._beta_increment_per_sampling = per_beta_increment_per_sampling
        self._max_abs_dist = max_abs_dist
        self._tree = SumTree(capacity)

    def _get_priority(self, dist: float) -> float:
        """

@param dist:
@return:
"""
        abs_dist = numpy.abs(dist) + self._epsilon

        if self._max_abs_dist:
            abs_dist = min(abs_dist, self._max_abs_dist)

        return abs_dist ** self._alpha

    def add(self, sample: Any, dist: float) -> None:
        """

@param sample:
@param error:
@return:
"""
        self._tree.push(sample, self._get_priority(dist))

    def sample(self, num: int) -> Iterable:
        """

@param num:
@return:
"""
        segment = self._tree.total / num
        data = []
        leaf_indices = []
        # priorities = []

        self._beta = numpy.min([1.0, self._beta + self._beta_increment_per_sampling])

        for i in range(num):
            (leaf_index, _, _, data_) = self._tree.get(
                random.uniform(segment * i, segment * (i + 1)), normalised_sum=False
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

        return data

    def update_last_batch(self, errors: Iterable) -> None:
        """

@param errors:
@return:
"""
        for leaf_index, error in zip(self._last_leaf_indices, errors):
            self.update(leaf_index, error)

    def update(self, leaf_index: int, error: float) -> None:
        """

@param leaf_index:
@param error:
@return:
"""
        self._tree.update_leaf(leaf_index, self._get_priority(error))

    def __len__(self) -> int:
        """

@return:
"""
        return len(self._tree)

    @property
    def capacity(self) -> int:
        return self._tree.capacity


if __name__ == "__main__":

    def stest_experience_buffer():
        capacity = 2 ** 8
        batch_size = 4

        rb = PrioritisedBuffer(capacity)
        for i in range(capacity):
            a = (i, i, i, i, i)
            rb.add(a, sum(a))

        print(rb.sample(batch_size))

        rb.update_last_batch(numpy.random.rand(capacity))

        print(rb.sample(batch_size))

    stest_experience_buffer()
