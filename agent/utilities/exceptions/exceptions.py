#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'


class HasNoEnvError(Exception):
  """
  Raised when an agent has no environment assigned and some implicit next or step called.
  """

  def __init__(self):
    msg = 'Agent has no env assigned'
    Exception.__init__(self, msg)


class NoTrainingProcedure(Exception):
  def __init__(self):
    msg = 'No TrainingProcedure'
    Exception.__init__(self, msg)


class NoTrajectoryException(Exception):
  def __init__(self):
    msg = 'No Trajectory Available'
    Exception.__init__(self, msg)
