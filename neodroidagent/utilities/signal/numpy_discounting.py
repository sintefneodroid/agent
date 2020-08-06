#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Iterable, Union

import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """

from scipy.signal import lfilter

__all__ = ["discount_signal", "discount_signal_numpy"]


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
