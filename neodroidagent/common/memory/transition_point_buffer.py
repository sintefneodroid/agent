#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count

from neodroidagent.common.memory.data_structures.expandable_circular_buffer import (
    ExpandableCircularBuffer,
)
from neodroidagent.common.memory.experience import Transition, TransitionPoint
from neodroidagent.utilities import NoData
from warg.arguments import wrap_args

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
  Buffer class for for maintaining a circular buffer of TransitionPoint's
"""
__all__ = ["TransitionPointBuffer"]


class TransitionPointBuffer(ExpandableCircularBuffer):
    def add_transition_points(self, transition_points: TransitionPoint) -> None:
        """
    Iteratively adds transition points with TransitionPointBuffer.add_transition_point

    @param transition_points:
    @return:
    """
        for t in zip(*transition_points):
            self.add_transition_point(TransitionPoint(*t))

    @wrap_args(TransitionPoint)
    def add_transition_point(self, transition_point: TransitionPoint) -> None:
        """
    args will be wrapped in a TransitionPoint type tuple and collected as transition_point

    @param transition_point:
    @return:
    """
        self._add(transition_point)

    def sample_transition_points(self, num=None) -> TransitionPoint:
        """Randomly sample transitions from memory."""
        if len(self):
            samples = self._sample(num)
            batch = TransitionPoint(*zip(*samples))
            return batch
        raise NoData


if __name__ == "__main__":
    tb = TransitionPointBuffer()
    print(TransitionPoint.get_fields().__len__())
    a = iter(count())
    for i in range(100):
        b = next(a)
        tp = TransitionPoint(*([b] * len(TransitionPoint.get_fields())))
        tb.add_transition_point(tp)
    print(tb.sample_transition_points(1))
