#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Iterable, Union

import numpy
import numpy as np
import torch
from numba import jit
from scipy.signal import lfilter

__author__ = 'cnheider'
__doc__ = ''


@jit(nopython=True, nogil=True)
def valued_discount(signal: np.ndarray,
                    next_estimate: np.ndarray,
                    terminal: np.ndarray,
                    discounting_factor: float):
  r"""
  Calculates discounted signal according to equation:

  .. math:: G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n})

  This function cares about episodes ends, so that if one row of the ``signal`` matrix contains multiple
  episodes
  it will use information from ``terminal`` to determine episode horizon.

  If the ``signal`` array contains unfinished episode this function will use values from
  ``next_estimate`` to
  calculate the :math:`\gamma^n V_{t+n-1}(S_{t+n})` term


  Legend for dimensions:
   * ``N`` - number of parallel agents
   * ``T`` - number of time steps

  :param signal: array of shape ``N*T`` containing rewards for each time step
  :param next_estimate: array of shape ``(N,)`` containing value estimates for last value(:math:`V_{
  t+n-1}`)
  :param terminal:  array of shape ``N*1`` containing information about episode ends
  :param discounting_factor: discount value(gamma)
  :return: array of shape ``N*T`` with discounted values for each step
  """

  v: np.ndarray = next_estimate
  discounted = numpy.zeros_like(signal)
  a = signal.shape[-1]
  for t in range(a - 1, -1, -1):
    r, termi = signal[:, t], terminal[:, t]
    v = (r + discounting_factor * v * (1. - termi)).astype(discounted.dtype)
    discounted[:, t] = v

  return discounted


def discount_signal(x: Union[numpy.ndarray, Iterable, int, float], factor) -> Any:
  """
      x = [r1, r2, r3, ..., rN]
      returns [r1 + r2*gamma + r3*gamma^2 + ...,
                 r2 + r3*gamma + r4*gamma^2 + ...,
                   r3 + r4*gamma + r5*gamma^2 + ...,
                      ..., ..., rN]


  # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
  # Here, we have y[t] - discount*y[t+1] = x[t]
  # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]


  C[i] = R[i] + discount * C[i+1]
  signal.lfilter(b, a, x, axis=-1, zi=None)
  a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                        - a[1]*y[n-1] - ... - a[N]*y[n-N]
  """

  a: Union[numpy.ndarray, Iterable, int, float] = lfilter([1],
                                                          [1, -factor],
                                                          numpy.flip(x, -1),
                                                          axis=-1)

  return numpy.flip(a, -1)

#@jit(nopython=True, nogil=True)
def discount_return(x, discount):
  return numpy.sum(x * (discount ** numpy.arange(len(x))))


if __name__ == '__main__':
  rollout = numpy.zeros((10,2)).T
  rollout[:,-1] = 1
  print(discount_signal(rollout, .5))
