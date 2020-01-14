#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroidagent.common.memory.data_structures.expandable_circular_buffer import (
    ExpandableCircularBuffer,
)
from neodroidagent.common.transitions.points import TrajectoryPoint
from warg.arguments import wrap_args

__author__ = "Christian Heider Nielsen"

__all__ = ["TrajectoryBuffer"]


class TrajectoryBuffer(ExpandableCircularBuffer):
    @wrap_args(TrajectoryPoint)
    def add_point(self, point):
        self._add(point)

    def retrieve_trajectory(self):
        if len(self):
            batch = TrajectoryPoint(*zip(*self._memory))
            return batch
        return [None] * TrajectoryPoint._fields.__len__()


if __name__ == "__main__":
    tb = TrajectoryBuffer()
