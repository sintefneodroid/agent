#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import numpy as np
from numba import jit
from scipy.signal import lfilter

__author__ = 'cnheider'
__doc__ = ''


@jit(nopython=True, nogil=True)
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


@jit(nopython=True, nogil=True)
def discount2(rewards: np.ndarray,
              estimate_of_last: np.ndarray,
              terminals: np.ndarray,
              discount: float):
  r"""
  Calculates discounted reward according to equation:

  .. math:: G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n})

  This function cares about episodes ends, so that if one row of the ``rewards`` matrix contains multiple
  episodes
  it will use information from ``dones`` to determine episode horizon.

  If the ``rewards`` array contains unfinished episode this function will use values from
  ``estimate_of_last`` to
  calculate the :math:`\gamma^n V_{t+n-1}(S_{t+n})` term

  *Note:* This function does not support n-step discounts calculation. For this functionality
          look at the Reward`/`Advantage` classes

  Legend for dimensions:
   * ``N`` - number of parallel agents
   * ``T`` - number of time steps

  :param rewards: array of shape ``N*T`` containing rewards for each time step
  :param estimate_of_last: array of shape ``(N,)`` containing value estimates for last value(:math:`V_{
  t+n-1}`)
  :param terminals:  array of shape ``N*1`` containing information about episode ends
  :param discount: discount value(gamma)
  :return: array of shape ``N*T`` with discounted values for each step
  """

  v: np.ndarray = estimate_of_last
  ret = np.zeros_like(rewards)
  for timestep in range(rewards.shape[1] - 1, -1, -1):
    r, done = rewards[:, timestep], terminals[:, timestep]
    v = (r + discount * v * (1. - done)).astype(ret.dtype)
    ret[:, timestep] = v
  return ret


if __name__ == '__main__':
  rollout = numpy.zeros((10))
  rollout[-1] = 1
  print(discount(rollout, .5))
