#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

import numpy

from draugr.torch_utilities import to_tensor

__author__ = "Christian Heider Nielsen"

import torch

__all__ = ["torch_advantage_estimate", "torch_compute_gae"]


def torch_advantage_estimate(
    signal,
    non_terminal,
    value_estimate,
    *,
    discount_factor: float = 0.95,
    tau: float = 0.97,
    device: Union[str, torch.device] = "cpu",
    normalise: bool = True,
    divide_by_zero_safety: float = 1e-10,
):
    """
Computes advantages and discounted returns.
If the advantage is positive for an action, then it yielded a more positive signal than expected. And thus
expectations can be adjust to make actions more likely.

:param discount_factor:
:type discount_factor:
:param tau:
:type tau:
:return:
:rtype:
@param device:
@param tau:
@param discount_factor:
@param value_estimate:
@param non_terminal:
@param signal:
@param divide_by_zero_safety:
@param normalise:
"""
    horizon_length, num_workers, *_ = signal.size()

    advantages_out = torch.zeros_like(signal, device=device)
    adv = torch.zeros(num_workers, device=device)

    for t in reversed(range(horizon_length - 1)):
        delta = (
            signal[t]
            + value_estimate[t + 1] * discount_factor * non_terminal[t]
            - value_estimate[t]
        )
        adv = adv * discount_factor * tau * non_terminal[t] + delta

        advantages_out[t] = adv

    if normalise:
        advantages_out = (advantages_out - advantages_out.mean()) / (
            advantages_out.std() + divide_by_zero_safety
        )

    return advantages_out


def torch_compute_gae(
    signal,
    non_terminal,
    values,
    *,
    discount_factor=0.95,
    gae_lambda=0.95,
    device: Union[str, torch.device] = "cpu",
    normalise_adv=True,
) -> torch.tensor:
    """

Computes discounted return and advantage

@param signal:
@param non_terminal:
@param values:
@param discount_factor:
@param gae_lambda:
@param device:
@param normalise:
@return:
"""
    len_signal = len(signal)
    assert len_signal == len(non_terminal) == len(values) - 1, (
        f"{signal.shape}, {non_terminal.shape}, " f"{values.shape}"
    )

    ret = []
    gae = 0
    for step_i in reversed(range(len_signal)):
        delta = (
            signal[step_i]
            + discount_factor * values[step_i + 1] * non_terminal[step_i]
            - values[step_i]
        )
        gae = delta + discount_factor * gae_lambda * non_terminal[step_i] * gae
        ret.insert(0, gae + values[step_i])

    ret = to_tensor(ret, device=device)
    advantage = ret - values[:-1]

    if normalise_adv:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

    return ret, advantage


if __name__ == "__main__":

    def s():

        numpy.random.seed(23)
        size = (10, 3, 1)
        a_size = (size[0] + 1, *size[1:])
        signal = numpy.zeros(size)
        non_terminal = numpy.ones(size)
        value_estimate = numpy.random.random(a_size)
        non_terminal[3, 0] = 0
        non_terminal[8, 1] = 0
        signal[-5:, :] = -1

        signals = to_tensor(signal, device="cpu")
        non_terminals = to_tensor(non_terminal, device="cpu")
        value_estimates = to_tensor(value_estimate, device="cpu")

        r, a = torch_compute_gae(signals, non_terminals, value_estimates)
        print(r, a)
        print(size, r.shape, a.shape)

    s()
