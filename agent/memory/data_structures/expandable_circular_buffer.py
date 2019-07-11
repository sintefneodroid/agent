#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

import numpy

__author__ = 'cnheider'

import random


class ExpandableCircularBuffer(object):
  '''For storing element in an expandable buffer.'''

  def __init__(self, capacity=0):
    self._capacity = capacity
    self._memory = []
    self._position = 0

  def _add(self, value):
    '''Adds value to memory'''
    if value is list:
      for val in value:
        self._add(val)
    else:
      if len(self._memory) < self._capacity or self._capacity == 0:
        self._memory.append(None)
      self._memory[self._position] = value
      self._position += 1
      if self._capacity != 0:
        self._position = self._position % self._capacity

  def _sample(self, req_num=None):
    '''Samples random values from memory'''
    if req_num is None:
      return numpy.random.shuffle(self._memory)
    else:
      num_entries = len(self._memory)

      if req_num > num_entries:
        logging.info(f'Buffer only has {num_entries},'
                        f' returning {num_entries} entries'
                        f' of the requested {req_num}')
        req_num = len(self._memory)

      batch = random.sample(self._memory, req_num)

      return batch

  def clear(self):
    del self._memory[:]
    self._position = 0

  def __len__(self):
    '''Return the length of the memory list.'''
    # return len(self._memory)
    return self._position
