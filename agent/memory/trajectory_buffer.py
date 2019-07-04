#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from agent.interfaces.specifications import TrajectoryPoint
from agent.memory import ExpandableCircularBuffer
from warg.arguments import namedtuple_args

__author__ = 'cnheider'


class TrajectoryBuffer(ExpandableCircularBuffer):

  @namedtuple_args(TrajectoryPoint)
  def add_point(self, point):
    self._add(point)

  def retrieve_trajectory(self):
    if len(self):
      batch = TrajectoryPoint(*zip(*self._memory))
      return batch
    return [None] * TrajectoryPoint._fields.__len__()
