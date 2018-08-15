#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utilities.sampling.bounded_triangle_sample import bounded_triangle_sample

__author__ = 'cnheider'

import numpy as np


def sample(set):
  if len(set):
    idx = np.random.randint(0, len(set))
    return set[idx]


if __name__ == '__main__':
  print(bounded_triangle_sample(np.arange(0, 10)))
