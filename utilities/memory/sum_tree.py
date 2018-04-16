#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'

class SumTree:

  def __init__(self, capacity):
    self._capacity = capacity
    self._updates = 0
    self._write = 0
    # self.tree = np.zeros(2 * capacity - 1)
    # self.data = np.zeros(capacity, dtype=object)
    self._tree = [0 for i in range(2 * capacity - 1)]
    self._data = [None for i in range(capacity)]

  def __propagate__(self, idx, change):
    parent = (idx - 1) // 2

    self._tree[parent] += change

    if parent != 0:
      self.__propagate__(parent, change)

  def _retrieve(self, idx, s):
    left = 2 * idx + 1
    right = left + 1

    if left >= len(self._tree):
      return idx

    if s <= self._tree[left]:
      return self._retrieve(left, s)
    else:
      return self._retrieve(right, s - self._tree[left])

  def total(self):
    return self._tree[0]

  def add(self, p, data):
    idx = self._write + self._capacity - 1

    self._data[self._write] = data
    self.update(idx, p)

    self._updates += 1
    self._write += 1
    if self._write >= self._capacity:
      self._write = 0

  def update(self, idx, p):
    change = p - self._tree[idx]

    self._tree[idx] = p
    self.__propagate__(idx, change)

  def get(self, s):
    idx = self._retrieve(0, s)
    dataIdx = idx - self._capacity + 1

    return (idx, self._tree[idx], self._data[dataIdx])

  def __len__(self):
    # return len(self.tree) #TODO: DOES NOT RETURN NUMBER OF ELEMENTS ADDED BUT TREE INDEX SIZE
    return self._updates

  def max_priority(self):
    return max(self._tree)
