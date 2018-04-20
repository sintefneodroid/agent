#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'


def get_upper_vars_of(module):
  v = vars(module)
  if v:
    return {key: value for key, value in module.__dict__.items() if key.isupper() or (key.startswith('_')                                                                                    and not key.endswith(
            '_'))}
  return {}