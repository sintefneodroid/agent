#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
import numpy as np

from utilities import S


class Summary:
  '''

'''

  def __init__(self):
    self._values = []
    self.length = 0
    self.running_mean = 0
    self.running_variance = 0

  def append(self, val):
    self._values.append(val)
    self.length += 1

  @property
  def values(self):
    return self._values

  def moving_average(self, window_size=100):
    if self.length >= window_size:
      return S.mean(self._values[-window_size:])
    elif self.length > 0:
      return S.mean(self._values)
    else:
      return 0

  def running_average(self, data):
    '''
Computes new running mean and variances.
:param data: New piece of data.
:return: New mean and variance values.
'''
    mean, var, steps = self.running_mean, self.running_variance, self.length
    current_x = np.mean(data, axis=0)

    new_mean = mean + (current_x - mean) / (steps + 1)
    new_variance = var + (current_x - new_mean) * (current_x - mean)

    self.running_mean, self.running_variance = new_mean, new_variance
    return new_mean, new_variance

  def __len__(self):
    return len(self._values)
