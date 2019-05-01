#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'

import torch

from agent import utilities as U


def advantage_estimate(signal,
                       non_terminal,
                       value_estimate,
                       *,
                       discount_factor=0.99,
                       tau=0.95,
                       device='cpu'
                       ):
  '''
    Computes advantages and discounted returns.
    If the advantage is positive for an action, then it yielded a more positive signal than expected. And thus
    expectations can be adjust to make actions more likely.

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

  signals = U.to_tensor(signal, device=device)
  non_terminals = U.to_tensor(non_terminal, device=device)
  value_estimates = U.to_tensor(value_estimate, device=device)

  T = signals.size()
  T = T[0]
  num_workers = 1
  # T,num_workers,_  = signals.size()

  advantages_out = torch.zeros(T, num_workers, 1).to(device)
  advantage_estimate_now = torch.zeros(num_workers, 1).to(device)

  for t in reversed(range(T - 1)):
    signal_now = signals[t]
    baseline_value_future = value_estimates[t + 1]
    baseline_value_now = value_estimates[t]
    non_terminal_now = non_terminals[t]

    td_error = signal_now + baseline_value_future * discount_factor * non_terminal_now - baseline_value_now
    advantage_estimate_now = advantage_estimate_now * discount_factor * tau * non_terminal_now + td_error

    advantages_out[t] = advantage_estimate_now

  advantages = advantages_out.squeeze()

  return advantages


def compute_gae(next_value,
                signals,
                masks,
                values,
                *,
                discount_factor=0.99,
                tau=0.95):
  with torch.no_grad():
    values = values + (next_value,)
    gae = 0.0
    returns = []
    for step in reversed(range(len(signals))):
      delta = signals[step] + discount_factor * values[step + 1] * masks[step] - values[step]
      gae = delta + discount_factor * tau * masks[step] * gae
      returns.insert(0, gae + values[step])
  return returns
