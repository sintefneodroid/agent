#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from agent.utilities.torch_utilities import to_tensor

__author__ = 'cnheider'

import torch


def torch_advantage_estimate(signal,
                             non_terminal,
                             value_estimate,
                             *,
                             discount_factor=0.95,
                             tau=0.95,
                             device='cpu',
                             normalise=True,
                             divide_by_zero_safety=1e-10
                             ):
  '''
    Computes advantages and discounted returns.
    If the advantage is positive for an action, then it yielded a more positive signal than expected. And thus
    expectations can be adjust to make actions more likely.

      :param value_estimate:
      :param non_terminal:
      :param signal:
      :param device:
    :param use_cuda:
    :type use_cuda:
    :param signals:
    :type signals:
    :param value_estimates:
    :type value_estimates:
    :param non_terminals:
    :type non_terminals:
    :param discount_factor:
    :type discount_factor:
    :param tau:
    :type tau:
    :return:
    :rtype:
  '''

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

    delta = signal_now + value_future * discount_factor * non_terminal_now - value_now
    adv = adv * discount_factor * tau * non_terminal_now + delta

    advantages_out[t] = adv

  if normalise:
    advantages_out = (advantages_out - advantages_out.mean()) / (advantages_out.std() + divide_by_zero_safety)

  return advantages_out


def torch_compute_gae(*,
                      signals,
                      non_terminals,
                      values,
                      #next_value,
                      discount_factor=0.95,
                      tau=0.95):
  with torch.no_grad():
    #values = values + (next_value,)
    gae = 0.0
    adv = []
    for step in reversed(range(len(signals))):
      delta = signals[step] + discount_factor * values[step + 1] * non_terminals[step] - values[step]
      gae = delta + discount_factor * tau * non_terminals[step] * gae
      adv.insert(0, gae + values[step])
  return adv
