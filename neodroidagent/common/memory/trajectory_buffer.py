#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroidagent.common.memory.data_structures.expandable_circular_buffer import (
    ExpandableCircularBuffer,
)
from neodroidagent.common.memory.experience.sample_points import SampleTrajectoryPoint
from neodroidagent.utilities import NoData
from warg.arguments import wrap_args

__author__ = "Christian Heider Nielsen"

__all__ = ["SampleTrajectoryBuffer"]


class SampleTrajectoryBuffer(ExpandableCircularBuffer):
    """
    Expandable buffer for storing rollout trajectories
  """

    @wrap_args(SampleTrajectoryPoint)
    def add_trajectory_point(self, point: SampleTrajectoryPoint):
        """

    @param point:
    @return:
    """
        self._add(point)

    def retrieve_trajectory(self) -> SampleTrajectoryPoint:
        """

    @return:
    """
        if len(self):
            batch = SampleTrajectoryPoint(*zip(*self._memory))
            return batch
        raise NoData


if __name__ == "__main__":
    tb = SampleTrajectoryBuffer()
    tb.retrieve_trajectory()
