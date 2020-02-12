#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Iterable, Union

import numpy
import torch
from numba import jit
from scipy.signal import lfilter

__author__ = "Christian Heider Nielsen"
__doc__ = ""

from draugr import to_tensor


@jit(nopython=True, nogil=True)
def valued_discount(
    signal: numpy.ndarray,
    next_estimate: numpy.ndarray,
    terminal: numpy.ndarray,
    discounting_factor: float,
):
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

    v: numpy.ndarray = next_estimate
    discounted = numpy.zeros_like(signal)
    a = signal.shape[-1]
    for t in range(a - 1, -1, -1):
        r, termi = signal[:, t], terminal[:, t]
        v = (r + discounting_factor * v * (1.0 - termi)).astype(discounted.dtype)
        discounted[:, t] = v

    return discounted
