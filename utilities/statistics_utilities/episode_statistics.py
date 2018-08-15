#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
import numpy as np


class EpisodeStatistics(object):

  def __init__(self):
    super().__init__()

  durations = []
  signals = []

  def moving_average(self, window_size=100):
    signal_ma = np.mean(self.signals[-window_size:])
    duration_ma = np.mean(self.durations[-window_size:])
    return signal_ma, duration_ma
