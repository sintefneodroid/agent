#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'

import numpy as np


def sample(set):
  if len(set):
    idx = np.random.randint(0, len(set))
    return set[idx]


def bounded_triangle_sample(set, mean=0.5, number=1):
  l = len(set)
  a = np.random.triangular(0, l * mean, l, number)
  a = int(np.floor(a)[0])

  return set[a]


if __name__ == '__main__':
  print(bounded_triangle_sample(np.arange(0, 10)))
