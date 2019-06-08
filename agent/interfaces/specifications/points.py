# !/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = 'cnheider'

from collections import namedtuple

TrajectoryPoint = namedtuple(
    'TrajectoryTrace', (
      'signal',
      'log_prob',
      'entropy',
      )
    )

AdvantageMemory = namedtuple(
    'AdvantageMemory', (
      'state',
      'action',
      'action_prob',
      'value_estimate',
      'advantage',
      'discounted_return',
      ),
    )
