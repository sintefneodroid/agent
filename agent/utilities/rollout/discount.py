#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy.signal import lfilter

__author__ = 'cnheider'
__doc__ = ''


def discount(x, gamma):
  """
      x = [r1, r2, r3, ..., rN]
      returns [r1 + r2*gamma + r3*gamma^2 + ...,
                 r2 + r3*gamma + r4*gamma^2 + ...,
                   r3 + r4*gamma + r5*gamma^2 + ...,
                      ..., ..., rN]
  """
  return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


if __name__ == '__main__':
  print(discount([0, 0, 0, 0, 0, 0, 0, 0, 1], .5))
