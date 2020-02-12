#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

import numpy

__author__ = "Christian Heider Nielsen"

import random

__all__ = ["ExpandableCircularBuffer"]

from neodroidagent.utilities import is_none_or_zero_or_negative


class ExpandableCircularBuffer(object):
    """
  For storing element in an expandable buffer.
  """

    def __init__(self, capacity: int = 0):
        """

    @param capacity:
    """
        self._capacity = capacity
        self._memory = []
        self._position = 0

    def _add(self, value):
        """Adds value to memory"""
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

    def _sample(self, num: int = None):
        """Samples random values from memory"""
        if self._capacity > 0:
            if is_none_or_zero_or_negative(num):
                a = self._memory
                numpy.random.shuffle(a)
                return a

            num_entries = len(self._memory)

            if num > num_entries:
                logging.info(
                    f"Buffer only has {num_entries},"
                    f" returning {num_entries} entries"
                    f" of the requested {num}"
                )
                num = len(self._memory)

            batch = random.sample(self._memory, num)

            return batch
        else:
            if num and num < len(self._memory):
                return self._memory[:num]
            return self._memory

    def clear(self):
        """

    @return:
    """
        del self._memory[:]
        self._position = 0

    def __len__(self):
        """Return the length of the memory list."""
        # return len(self._memory)
        return self._position


if __name__ == "__main__":
    unbounded = ExpandableCircularBuffer()
    for i in range(100):
        unbounded._add(i)
    print(unbounded._sample())
    print(unbounded._sample(4))

    bounded = ExpandableCircularBuffer(100)
    for i in range(200):
        bounded._add(i)
    print(bounded._sample())
    print(bounded._sample(4))
