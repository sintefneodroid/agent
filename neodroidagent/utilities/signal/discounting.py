#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union, Iterable, Any

import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """

import torch
from numba import jit

from scipy.signal import lfilter

__all__ = ["discount_signal", "discount_signal_numpy", "discount_signal_torch"]

from draugr import to_tensor, global_torch_device


# @jit(nopython=True, nogil=True)
def discount_signal(signal: list, discounting_factor: float) -> list:
    """

  @param signal:
  @param discounting_factor:
  @return:
  """
    signals = []
    r_ = numpy.zeros_like(signal[0])
    for r in signal[::-1]:
        r_ = r + discounting_factor * r_
        signals.insert(0, r_)
    return signals


# @jit(nopython=True, nogil=True)
def discount_signal_numpy(
    signal: Union[numpy.ndarray, Iterable, int, float], discounting_factor: float
) -> numpy.ndarray:
    """
signal = [s_1, s_2, s_3 ..., s_N]
returns [s_1 + s_2*discounting_factor + s_3*discounting_factor^2 + ...,
           s_2 + s_3*discounting_factor + s_4*discounting_factor^2 + ...,
             s_3 + s_4*discounting_factor + s_5*discounting_factor^2 + ...,
                ..., ..., s_N]


# See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
# Here, we have y[t] - discount*y[t+1] = x[t]
# or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]

C[i] = R[i] + discount * C[i+1]
signal.lfilter(b, a, x, axis=-1, zi=None)
a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                  - a[1]*y[n-1] - ... - a[N]*y[n-N]
"""

    # return numpy.sum(signal * (discounting_factor ** numpy.arange(len(signal))))

    a: Union[numpy.ndarray, Iterable, int, float] = lfilter(
        [1], [1, -discounting_factor], numpy.flip(signal, -1), axis=-1
    )

    return numpy.flip(a, -1)


# @jit(nopython=True, nogil=True)
def discount_signal_torch(
    signal: torch.Tensor,
    discounting_factor: float,
    *,
    device=global_torch_device(),
    non_terminal=None
) -> Any:
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

  non_terminal if supplied lets you define a mask for masking terminal state signals.


  @param signal:
  @param discounting_factor:
  @param device:
  @param non_terminal:
  @return:
  """

    discounted = torch.zeros_like(signal, device=device)
    R = torch.zeros(signal.shape[0], device=device)
    NT = torch.ones_like(signal, device=device)
    if non_terminal is not None:
        NT = to_tensor(non_terminal, device=device)
    for i in reversed(range(signal.shape[1])):
        R = signal[:, i] + discounting_factor * R * NT[:, i]
        discounted[:, i] = R

    return discounted


if __name__ == "__main__":

    def s():
        rollout = numpy.zeros((10, 2)).T
        rollout_nt = numpy.ones((10, 2)).T
        rollout_nt[0, 3] = 0
        rollout_nt[1, 8] = 0
        rollout[:, -5:] = -1
        print(discount_signal(rollout[0], 0.5))

        print(discount_signal_numpy(rollout, 0.5))

        print(
            discount_signal_torch(to_tensor(rollout, device="cpu"), 0.5, device="cpu")
        )

        print(
            discount_signal_torch(
                to_tensor(rollout, device="cpu"),
                0.5,
                device="cpu",
                non_terminal=rollout_nt,
            )
        )

    s()
