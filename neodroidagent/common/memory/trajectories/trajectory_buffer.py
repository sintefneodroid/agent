#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Any, Sized

from neodroidagent.common.memory.data_structures.expandable_circular_buffer import (
    ExpandableCircularBuffer,
)
from neodroidagent.utilities import NoData
from warg import IndexDictTuplesMixin, IterDictValuesMixin
from warg.arguments import wrap_args

__author__ = "Christian Heider Nielsen"

__all__ = ["SampleTrajectoryBuffer", "SampleTrajectoryPoint", "SamplePoint"]


@dataclass
class SamplePoint(IterDictValuesMixin, IndexDictTuplesMixin):
    """
__slots__=["action", "distribution"]
"""

    action: Any
    distribution: Any

    @staticmethod
    def get_fields() -> Sized:
        """

@return:
"""
        return SamplePoint.__slots__

    def __len__(self):
        """

@return:
"""
        return len(self.action)


@dataclass
class SampleTrajectoryPoint(IterDictValuesMixin, IndexDictTuplesMixin):
    """
__slots__=["signal", "terminated", "action", "distribution"]
"""

    signal: Any
    terminated: Any
    action: Any
    distribution: Any

    def __len__(self):
        """

@return:
"""
        return len(self.signal)


class SampleTrajectoryBuffer(ExpandableCircularBuffer):
    """
Expandable buffer for storing rollout trajectories
"""

    def __init__(self):
        super().__init__()

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
            return SampleTrajectoryPoint(*zip(*self._sample()))
        raise NoData


if __name__ == "__main__":
    tb = SampleTrajectoryBuffer()
    NoneAr = list(range(6))
    [tb.add_trajectory_point(NoneAr, NoneAr, NoneAr, NoneAr) for _ in range(10)]
    print(tb.retrieve_trajectory())
