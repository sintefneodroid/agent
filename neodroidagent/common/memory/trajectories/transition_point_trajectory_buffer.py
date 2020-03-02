#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count

from neodroidagent.common.memory.data_structures.expandable_circular_buffer import (
    ExpandableCircularBuffer,
)
from neodroidagent.common.memory.transitions import ValuedTransitionPoint
from neodroidagent.utilities import NoData
from warg.arguments import wrap_args

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
  Buffer class for for maintaining a circular buffer of TransitionPoint's
"""
__all__ = ["TransitionPointTrajectoryBuffer"]


class TransitionPointTrajectoryBuffer(ExpandableCircularBuffer):
    def __init__(self):
        super().__init__()

    @wrap_args(ValuedTransitionPoint)
    def add_transition_point(self, transition_point: ValuedTransitionPoint) -> None:
        """
args will be wrapped in a TransitionPoint type tuple and collected as transition_point

@param transition_point:
@return:
"""
        self._add(transition_point)

    def sample(self) -> ValuedTransitionPoint:
        """Randomly sample transitions from memory."""
        if len(self):
            return ValuedTransitionPoint(*zip(*self._sample()))
        raise NoData


if __name__ == "__main__":
    tb = TransitionPointTrajectoryBuffer()
    print(ValuedTransitionPoint.get_fields().__len__())
    a = iter(count())
    for i in range(100):
        b = next(a)
        tp = ValuedTransitionPoint(*([b] * len(ValuedTransitionPoint.get_fields())))
        tb.add_transition_point(tp)
    print(tb.sample(10))
