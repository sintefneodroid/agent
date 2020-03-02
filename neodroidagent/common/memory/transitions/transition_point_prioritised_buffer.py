#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any

import numpy

__author__ = "Christian Heider Nielsen"
__all__ = ["TransitionPointPrioritisedBuffer"]
__doc__ = r"""
  The main idea is that we prefer transitions that does not fit well to our current estimate of the Q
  function, because these are the transitions that we can learn most from. This reflects a simple intuition
  from our real world - if we encounter a situation that really differs from our expectation,
  we think about it over and over and change our model until it fits.
  """

from neodroidagent.common.memory.data_structures import PrioritisedBuffer
from neodroidagent.common.memory.transitions.transitions import TransitionPoint


class TransitionPointPrioritisedBuffer(PrioritisedBuffer):
    """

"""

    def add_transition_point(self, sample: Any, error: float = 0.0) -> None:
        super().add(sample, error)

    def sample(self, num: int) -> Any:
        return TransitionPoint(*zip(*super().sample(num)))


if __name__ == "__main__":

    def stest_experience_buffer():
        capacity = 2 ** 8
        batch_size = 4

        rb = TransitionPointPrioritisedBuffer(capacity)
        for i in range(capacity):
            a = TransitionPoint(i, i, i, i, i)
            rb.add_transition_point(a, sum(a))

        print(rb.sample(batch_size))

        rb.update_last_batch(numpy.random.rand(capacity))

        print(rb.sample(batch_size))

    stest_experience_buffer()
