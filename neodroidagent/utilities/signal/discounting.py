#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Iterable, Union

import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """

import torch

from scipy.signal import lfilter

__all__ = ["discount_rollout_signal_torch"]

from draugr.torch_utilities import to_tensor, global_torch_device


# @jit(nopython=True, nogil=True)
def discount_rollout_signal_torch(
    signal: torch.Tensor,
    discounting_factor: float,
    *,
    device=global_torch_device(),
    non_terminal=None,
    batch_normalised=False,
    epsilon=1e-3
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
    @return:"""

    discounted = torch.zeros_like(signal, device=device)
    R = torch.zeros(signal.shape[-1], device=device)
    NT = torch.ones_like(signal, device=device)
    if non_terminal is not None:
        NT = to_tensor(non_terminal, device=device)

    for i in reversed(range(signal.shape[0])):
        R = signal[i] + discounting_factor * R * NT[i]
        discounted[i] = R

    if batch_normalised:
        # WARNING! Sometimes causes NANs!
        discounted = (discounted - discounted.mean()) / (discounted.std() + epsilon)

    return discounted


if __name__ == "__main__":

    def s():
        shapes = (10, 2, 1)
        signal = numpy.zeros(shapes)
        rollout_nt = numpy.ones(shapes)
        rollout_nt[3, 0] = 0
        rollout_nt[8, 1] = 0
        signal[-5:] = 1

        print("\n")
        print(
            discount_rollout_signal_torch(
                to_tensor(signal, device="cpu"), 0.5, device="cpu"
            )
        )

        print("\n")
        print(
            discount_rollout_signal_torch(
                to_tensor(signal, device="cpu"),
                0.5,
                device="cpu",
                non_terminal=rollout_nt,
            )
        )

    def ssadasf():
        shapes = (100, 2, 1)
        signal = numpy.ones(shapes)
        rollout_nt = numpy.ones(shapes)
        # rollout_nt[3, 0] = 0
        # rollout_nt[8, 1] = 0
        # signal[-1] = 0

        print("\n")
        print(
            discount_rollout_signal_torch(
                to_tensor(signal, device="cuda"), 0.5, device="cuda"
            )
        )

        print("\n")
        print(
            discount_rollout_signal_torch(
                to_tensor(signal, device="cpu"),
                0.5,
                device="cpu",
                non_terminal=rollout_nt,
            )
        )

    def zeroes_ssadasf():
        shapes = (100, 2, 1)
        signal = numpy.zeros(shapes)
        rollout_nt = numpy.ones(shapes)
        # rollout_nt[3, 0] = 0
        # rollout_nt[8, 1] = 0
        # signal[-1] = 0

        print("\n")
        print(
            discount_rollout_signal_torch(
                to_tensor(signal, device="cuda"), 0.5, device="cuda"
            )
        )

        print("\n")
        print(
            discount_rollout_signal_torch(
                to_tensor(signal, device="cpu"),
                0.5,
                device="cpu",
                non_terminal=rollout_nt,
            )
        )

    def small_ssadasf():
        shapes = (100, 2, 1)
        signal = numpy.ones(shapes) * 52e-16
        rollout_nt = numpy.ones(shapes)
        # rollout_nt[3, 0] = 0
        # rollout_nt[8, 1] = 0
        # signal[-1] = 0

        print("\n")
        print(
            discount_rollout_signal_torch(
                to_tensor(signal, device="cuda"), 0.5, device="cuda"
            )
        )

        print("\n")
        print(
            discount_rollout_signal_torch(
                to_tensor(signal, device="cpu"),
                0.5,
                device="cpu",
                non_terminal=rollout_nt,
            )
        )

    def small_ssadasf_negative():
        shapes = (100, 2, 1)
        signal = -numpy.ones(shapes) * 52e-16
        rollout_nt = numpy.ones(shapes)
        # rollout_nt[3, 0] = 0
        # rollout_nt[8, 1] = 0
        # signal[-1] = 0

        print("\n")
        print(
            discount_rollout_signal_torch(
                to_tensor(signal, device="cuda"), 0.5, device="cuda"
            )
        )

        print("\n")
        print(
            discount_rollout_signal_torch(
                to_tensor(signal, device="cpu"),
                0.5,
                device="cpu",
                non_terminal=rollout_nt,
            )
        )

    def ssadasf_negative():
        shapes = (100, 2, 1)
        signal = -numpy.ones(shapes)
        rollout_nt = numpy.ones(shapes)
        # rollout_nt[3, 0] = 0
        # rollout_nt[8, 1] = 0
        # signal[-1] = 0

        print("\n")
        print(
            discount_rollout_signal_torch(
                to_tensor(signal, device="cuda"), 0.99, device="cuda"
            )
        )

        print("\n")
        print(
            discount_rollout_signal_torch(
                to_tensor(signal, device="cpu"),
                0.99,
                device="cpu",
                non_terminal=rollout_nt,
            )
        )

    # s()
    # ssadasf()
    # zeroes_ssadasf()
    # small_ssadasf()
    # small_ssadasf_negative()
    ssadasf_negative()
