#!/usr/bin/env python3
# coding=utf-8
from warnings import warn

__author__ = 'cnheider'
import statistics as S
import utilities as U


class StatisticAggregator(object):

  def __init__(self, measures=S.__all__[1:], keep_measure_history=False):
    self._values = []
    self._length = 0

    # for key in self._measure_keys:
    #  setattr(self,key,None)

    self._measure_keys = measures
    self._keep_measure_history = keep_measure_history
    if self._keep_measure_history:
      self._measures = {}
      for key in self._measure_keys:
        self._measures[key] = []

  @property
  def values(self):
    return self._values

  @property
  def max(self):
    return max(self._values)

  @property
  def min(self):
    return min(self._values)

  @property
  def measures(self):
    if self._keep_measure_history:
      return self._measures
    else:
      out = {}
      for key in self._measure_keys:
        try:
          val = getattr(S, key)(self._values)
        except S.StatisticsError as e:
          warn(f'{e}')
          val = None
        out[key] = val
      return out

  def add(self, values):
    self.append(values)

  def append(self, values):
    self._values.append(values)
    if type is list:
      self._length += len(values)
    else:
      self._length += 1

    if self._keep_measure_history:
      for key in self._measure_keys:
        if self._length > 1:
          try:
            val = getattr(S, key)(self._values)
          except:
            val = None
          self._measures[key].append(val)
        else:
          warn(f'Length of statistical values are <=1, measure "{key}" maybe ill-defined')
          try:
            val = getattr(S, key)(self._values)
          except S.StatisticsError as e:
            warn(f'{e}')
            val = None
          self._measures[key].append(val)

  def __getattr__(self, item):
    if item in self._measure_keys:
      if self._length > 1:
        if self._keep_measure_history:
          return self._measures[item]
        else:
          try:
            return getattr(S, item)(self._values)
          except S.StatisticsError as e:
            warn(f'{e}')
            return None
      else:
        warn(f'Length of statistical values are <=1, measure "{item}" maybe ill-defined')
        try:
          return getattr(S, item)(self._values)
        except S.StatisticsError as e:
          warn(f'{e}')
          return None
    else:
      raise AttributeError

  def __repr__(self):
    return f'<StatisticAggregator> values: { self._values }, measures: {self.measures} </StatisticAggregator>'

  def __str__(self):
    return str(self._values)

  def __len__(self):
    return len(self._values)

  def moving_average(self, window_size=100):
    if self._length >= window_size:
      return S.mean(self._values[-window_size:])
    elif self._length > 0:
      return S.mean(self._values)
    else:
      return 0

  def save(self, file_name, **kwargs):
    U.save_statistic(self._values, file_name, **kwargs)


if __name__ == '__main__':
  signals = StatisticAggregator(keep_measure_history=False)

  for i in range(10):
    signals.append(i)

  print(signals)
  print(signals.measures)
  print(signals.variance)
  print(signals.moving_average())
  print(signals.max)
  print(signals.min)
