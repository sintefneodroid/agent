#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroidagent.common.memory.data_structures.expandable_circular_buffer import (
    ExpandableCircularBuffer,
)
from neodroidagent.common.memory.experience import (
    Transition,
    TransitionPoint,
    SampleTransitionPoint,
)
from neodroidagent.utilities import NoData
from warg.arguments import wrap_args

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
  Buffer class for for maintaining a circular buffer of TransitionPoint's
"""
__all__ = ["SampleTransitionBuffer"]


class SampleTransitionBuffer(ExpandableCircularBuffer):
    def add_transition_points(self, transition_points: SampleTransitionPoint) -> None:
        """
    Iteratively adds transition points with TransitionPointBuffer.add_transition_point

    @param transition_points:
    @return:
    """
        for t in zip(*transition_points):
            self.add_transition_point(SampleTransitionPoint(*t))

    @wrap_args(SampleTransitionPoint)
    def add_transition_point(self, transition_point: SampleTransitionPoint) -> None:
        """
    args will be wrapped in a TransitionPoint type tuple and collected as transition_point

    @param transition_point:
    @return:
    """
        self._add(transition_point)

    def sample_transition_points(self, num=None) -> SampleTransitionPoint:
        """Randomly sample transitions from memory."""
        if len(self):
            samples = self._sample(num)
            batch = SampleTransitionPoint(*zip(*samples))
            return batch
        raise NoData


if __name__ == "__main__":
    tb = SampleTransitionBuffer()
    print(Transition.get_fields().__len__())
    print(tb.sample_transition_points(5))
    tb.add_transition_point(None, None, None, None, None)
    print(tb.sample_transition_points(1))
