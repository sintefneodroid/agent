#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

from collections import namedtuple

TrajectoryTrace = namedtuple(
    'TrajectoryTrace', (
      'signal',
      'log_prob',
      'entropy',
      )
    )

ValuedTransition = namedtuple(
    'ValuedTransition', (
      'state',
      'action',
      'action_prob',
      'value_estimate',
      'signal',
      'successor_state',
      'non_terminal',
      ),
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

Transition = namedtuple(
    'Transition', (
      'state',
      'action',
      'signal',
      'successor_state',
      'non_terminal'
      )
    )
