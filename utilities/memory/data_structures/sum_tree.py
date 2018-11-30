#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

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


class SumTree2(object):

  def __init__(self, max_size):
    self.max_size = max_size
    self.tree_level = int(math.ceil(math.log(max_size + 1, 2))) + 1
    self.tree_size = int(2 ** self.tree_level - 1)
    self.tree = [0 for _ in range(self.tree_size)]
    self.data = [None for _ in range(self.max_size)]
    self.size = 0
    self.cursor = 0

  def add(self, contents, value):
    index = self.cursor
    self.cursor = (self.cursor + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

    self.data[index] = contents
    self.val_update(index, value)

  def get_val(self, index):
    tree_index = 2 ** (self.tree_level - 1) - 1 + index
    return self.tree[tree_index], tree_index

  def val_update(self, index, value):
    old_value, tree_index = self.get_val(index)
    diff = value - old_value
    self.reconstruct(tree_index, diff)

  def reconstruct(self, tindex, diff):
    self.tree[tindex] += diff
    if not tindex == 0:
      tindex = int((tindex - 1) / 2)
      self.reconstruct(tindex, diff)

  def find(self, value, norm=True):
    if norm:
      value *= self.tree[0]
    return self._find(value, 0)

  def _find(self, value, index):
    i = 2 ** (self.tree_level - 1) - 1
    if i <= index:
      idx = index - i
      return self.data[idx], self.tree[index], idx

    left = self.tree[2 * index + 1]

    if value <= left:
      return self._find(value, 2 * index + 1)
    else:
      return self._find(value - left, 2 * (index + 1))

  def print_tree(self):
    for k in range(1, self.tree_level + 1):
      for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
        print(self.tree[j], end=' ')
      print()

  def filled_size(self):
    return self.size
