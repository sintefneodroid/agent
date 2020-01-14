#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroidagent.common.memory.data_structures.expandable_circular_buffer import (
    ExpandableCircularBuffer,
)
from neodroidagent.common.transitions.transitions import Transition
from warg.arguments import wrap_args

__author__ = "Christian Heider Nielsen"

__all__ = ["TransitionBuffer"]


class TransitionBuffer(ExpandableCircularBuffer):
    def add_transitions(self, transitions):
        """
    Iteratively adds transitions with TransitionBuffer.add_transition

    @param transitions:
    @return:
    """

        for t in transitions:
            self.add_transition(t)

    @wrap_args(Transition)
    def add_transition(self, transition):
        """
    args will be wrapped in a Transition type tuple and collected as transition

    @param transition:
    @return:
    """
        self._add(transition)

    def sample_transitions(self, num):
        """Randomly sample transitions from memory."""
        if len(self):
            batch = Transition(*zip(*self._sample(num)))
            return batch
        return [None] * Transition.get_fields().__len__()


if __name__ == "__main__":
    tb = TransitionBuffer()
    print(Transition.get_fields().__len__())
    print(tb.sample_transitions(5))
    tb.add_transition(None, None, None, None, None)
    print(tb.sample_transitions(1))
