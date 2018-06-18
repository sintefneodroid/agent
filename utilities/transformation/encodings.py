#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

import numpy as np


def signed_ternary_encoding(size: int, index):
  a = np.zeros(size)
  if index < size:
    a[index] = 1
  else:
    a[index - size] = -1
  return a


def to_one_hot(dims, index):
  if isinstance(index, np.int) or isinstance(index, np.int64):
    one_hot = np.zeros(dims)
    one_hot[index] = 1.
  else:
    one_hot = np.zeros((len(index), dims))
    one_hot[np.arange(len(index)), index] = 1.
  return one_hot


def agg_double_list(l):
  # l: [ [...], [...], [...] ]
  # l_i: result of each step in the i-th episode
  s = [np.sum(np.array(l_i), 0) for l_i in l]
  s_mu = np.mean(np.array(s), 0)
  s_std = np.std(np.array(s), 0)
  return s_mu, s_std
