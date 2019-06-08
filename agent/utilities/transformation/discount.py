#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
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
  return numpy.flip(lfilter([1],
                            [1, -gamma],
                            numpy.flip(x, -1),
                            axis=-1),
                    -1)


if __name__ == '__main__':
  rollout = numpy.zeros((10))
  rollout[-1] = 1
  print(discount(rollout, .5))
