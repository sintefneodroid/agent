#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from agent.utilities.signal.experimental.discounting import valued_discount

__author__ = 'cnheider'
__doc__ = ''


def discounted_ge(signals: np.ndarray,
                  values: np.ndarray,
                  terminals: np.ndarray,
                  discount_factor: float,
                  step_factor: float) -> np.ndarray:
  return discounted_gae(signals=signals,
                        values=values,
                        terminals=terminals,
                        discount_factor=discount_factor,
                        step_factor=step_factor) + values[:, :-1]


def discounted_gae(*,
                   signals: np.ndarray,
                   values: np.ndarray,
                   terminals: np.ndarray,
                   discount_factor: float,
                   step_factor: float) -> np.ndarray:
  """

  :param terminals:
  :param values:
  :param signals:
  :param discount_factor: the discount factor as we know it from n-step rewards
  :param step_factor: can be interpreted as the `n` in n-step rewards. Specifically setting it to 0
  reduces the  equation to be single step TD error, while setting it to 1 means there is no horizon
  so estimate over all steps
  """

  td_errors = signals + discount_factor * values[:, 1:] * (1. - terminals) - values[:, :-1]
  return valued_discount(td_errors,
                         np.zeros_like(values[:, 0]),
                         terminals,
                         step_factor * discount_factor)
