#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
from collections import namedtuple

from typing import Callable

__author__ = 'cnheider'
__doc__ = ''

# class LearningConfig(object):
#  pass

# class EnvironmentConfig(object):
#  pass

TrainingSpecification = namedtuple('TrainingSpecification',
                                   ['constructor',
                                    'kwargs'])


class TrainingSession(abc.ABC):
  def __init__(self, *, training_procedure: Callable, **kwargs):
    self._training_procedure = training_procedure
    pass

  def __call__(self, *args, **kwargs):
    '''
    Start an Agent training session, builds Agent and starts/connect environment(s)


    :param args:
    :param kwargs:
    :return:
    '''

    pass
