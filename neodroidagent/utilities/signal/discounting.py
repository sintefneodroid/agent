#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 09/10/2019
           '''


def discount_signal(self, signals, value):
  discounted_r = numpy.zeros_like(signals)
  running_add = value
  for t in reversed(range(0, len(signals))):
    running_add = running_add * self.gamma + signals[t]
    discounted_r[t] = running_add
  return discounted_r
