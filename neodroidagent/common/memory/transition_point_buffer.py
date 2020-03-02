#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count

from neodroidagent.common.memory.data_structures.expandable_circular_buffer import (
    ExpandableCircularBuffer,
)
from neodroidagent.common.memory.transitions import TransitionPoint
from neodroidagent.utilities import NoData
from warg.arguments import wrap_args

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
  Buffer class for for maintaining a circular buffer of TransitionPoint's
"""
__all__ = ["TransitionPointBuffer"]


class TransitionPointBuffer(ExpandableCircularBuffer):
    @wrap_args(TransitionPoint)
    def add_transition_point(self, transition_point: TransitionPoint) -> None:
        """
args will be wrapped in a TransitionPoint type tuple and collected as transition_point

@param transition_point:
@return:
"""
        self._add(transition_point)

    def sample(self, num) -> TransitionPoint:
        """Randomly sample transitions from memory."""
        if len(self):
            return TransitionPoint(*zip(*self._sample(num)))
        raise NoData


if __name__ == "__main__":
    tb = TransitionPointBuffer(10)
    print(TransitionPoint.get_fields().__len__())
    a = iter(count())
    for i in range(21):
        b = next(a)
        tp = TransitionPoint(*([b] * len(TransitionPoint.get_fields())))
        tb.add_transition_point(tp)

    print(tb.sample(9))
