#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    tau: float = 0.95,
    device: str = "cpu",
    normalise: bool = True,
    divide_by_zero_safety: float = 1e-10
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

    signals = to_tensor(signal, device=device)
    non_terminals = to_tensor(non_terminal, device=device)
    value_estimates = to_tensor(value_estimate, device=device)

    if len(signals.size()) > 1:
        horizon_length, num_workers, *_ = signals.size()
    else:
        horizon_length = signals.size()[0]
        num_workers = 1

    advantages_out = torch.zeros(horizon_length, num_workers, 1, device=device)
    adv = torch.zeros(num_workers, 1, device=device)

    for t in reversed(range(horizon_length - 1)):
        signal_now = signals[t]
        value_future = value_estimates[t + 1]
        value_now = value_estimates[t]
        non_terminal_now = non_terminals[t]

        delta = (
            signal_now + value_future * discount_factor * non_terminal_now - value_now
        )
        adv = adv * discount_factor * tau * non_terminal_now + delta

        advantages_out[t] = adv

    if normalise:
        advantages_out = (advantages_out - advantages_out.mean()) / (
            advantages_out.std() + divide_by_zero_safety
        )

    return advantages_out


def torch_compute_gae(
    *, signals, non_terminals, values, discount_factor=0.95, tau=0.95
):
    signals = signals[:-1]
    non_terminals = non_terminals[:-1]
    with torch.no_grad():
        adv = []
        td_i = 0
        for step in reversed(range(len(signals))):
            td_i = (
                signals[step]
                + discount_factor * values[step + 1] * non_terminals[step]
                + discount_factor * tau * td_i
            )
            adv.append(td_i)

    adv.reverse()
    return adv


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
