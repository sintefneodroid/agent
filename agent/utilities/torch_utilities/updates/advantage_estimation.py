#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
from scipy.signal import lfilter

__author__ = 'cnheider'

import torch

from agent import utilities as U


def advantage_estimate(signal,
                       non_terminal,
                       value_estimate,
                       *,
                       discount_factor=0.95,
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


def compute_gae(*,
                signals,
                masks,
                values,
                next_value,
                discount_factor=0.95,
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


def discount_cumsum(x, discount):
  # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
  # Here, we have y[t] - discount*y[t+1] = x[t]
  # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
  return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def discount_return(x, discount):
  return numpy.sum(x * (discount ** numpy.arange(len(x))))


def compute_gae3(*,
                 signals,
                 masks,
                 values,
                 next_value,
                 discount_factor=0.95,
                 tau=0.95):
  with torch.no_grad():
    rews = numpy.append(signals, next_value)
    vals = numpy.append(values, next_value)

    # the next two lines implement GAE-Lambda advantage calculation
    deltas = rews[:-1] + discount_factor * vals[1:] - vals[:-1]
    GAELambdaAdv = discount_cumsum(deltas, discount_factor * tau)

    # the next line computes rewards-to-go, to be targets for the value function
    signals_to_go = discount_cumsum(rews, discount_factor)[:-1]


def compute_gae2(rewards, values, bootstrap_values, terminals, gamma, lam):
  # (N, T) -> (T, N)
  rewards = numpy.transpose(rewards, [1, 0])
  values = numpy.transpose(values, [1, 0])
  values = numpy.vstack((values, [bootstrap_values]))
  terminals = numpy.transpose(terminals, [1, 0])
  # compute delta
  deltas = []
  for i in reversed(range(rewards.shape[0])):
    V = rewards[i] + (1.0 - terminals[i]) * gamma * values[i + 1]
    delta = V - values[i]
    deltas.append(delta)
  deltas = np.array(list(reversed(deltas)))
  # compute gae
  A = deltas[-1, :]
  advantages = [A]
  for i in reversed(range(deltas.shape[0] - 1)):
    A = deltas[i] + (1.0 - terminals[i]) * gamma * lam * A
    advantages.append(A)
  advantages = reversed(advantages)
  # (T, N) -> (N, T)
  advantages = numpy.transpose(list(advantages), [1, 0])
  return advantages
