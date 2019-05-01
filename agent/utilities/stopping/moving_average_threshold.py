#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
from functools import partial


def ma_threshold(ma, solved_threshold=10):
  return ma >= solved_threshold


def ma_stop(solved_threshold=10):
  return partial(ma_threshold, solved_threshold=solved_threshold)


if __name__ == '__main__':
  stopping_condition = ma_stop(10)

  print(stopping_condition(1))
  print(stopping_condition(11))
