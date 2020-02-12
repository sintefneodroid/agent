#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = ""

import numpy

from neodroidagent.utilities.signal.experimental.discounting import valued_discount


def discounted_nstep(
    signals: numpy.ndarray,
    values: numpy.ndarray,
    terminals: numpy.ndarray,
    discount_factor,
    n,
) -> numpy.ndarray:
    r"""
Implementation of n-step reward given by the equation:

.. math:: G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n})


:param discount_factor: discount value. Should be between `(0, 1]`
:param n: (optional) number of steps to compute reward over. If `None` then calculates it till
   the end of episode
"""

    if n is None:
        return valued_discount(signals, values[:, -1], terminals, discount_factor)

    discounted = numpy.zeros_like(signals)

    for start in range(signals.shape[1]):
        end = min(start + n, signals.shape[1])
        discounted[:, start] = valued_discount(
            signals[:, start:end],
            values[:, end],
            terminals[:, start:end],
            discount_factor,
        )[:, 0]
    return discounted


def discounted_nstep_adv(
    signals: numpy.ndarray,
    values: numpy.ndarray,
    terminals: numpy.ndarray,
    discount_factor,
    n=None,
) -> numpy.ndarray:
    r"""
Implementation of n-step advantage given by the equation:

.. math:: \hat{A}_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V_{
t+n-1}(S_{t+n})
- V_{t+n-1}(S_{t+1})

"""
    return (
        discounted_nstep(signals, values, terminals, discount_factor, n)
        - values[:, :-1]
    )
