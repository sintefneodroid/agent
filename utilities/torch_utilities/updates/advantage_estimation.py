#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'

import numpy as np
import torch

import utilities as U


def advantage_estimate(
    signal,
    non_terminal,
    value_estimate,
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


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
  values = values + (next_value,)
  gae = 0
  returns = []
  for step in reversed(range(len(rewards))):
    delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
    gae = delta + gamma * tau * masks[step] * gae
    returns.insert(0, gae + values[step])
  return returns


'''
processed_rollout = [None] * (len(rollout) - 1)
advantages = self.network.tensor(np.zeros((config.num_workers, 1)))
returns = pending_value.data
for i in reversed(range(len(rollout) - 1)):
    states, value, actions, log_probs, rewards, terminals = rollout[i]
    terminals = self.network.tensor(terminals).unsqueeze(1)
    rewards = self.network.tensor(rewards).unsqueeze(1)
    actions = self.network.variable(actions)
    states = self.network.variable(states)
    next_value = rollout[i + 1][1]
    returns = rewards + config.discount * terminals * returns
    if not config.use_gae:
        advantages = returns - value.data
    else:
        td_error = rewards + config.discount * terminals * next_value.data - value.data
        advantages = advantages * config.gae_tau * config.discount * terminals + td_error
    processed_rollout[i] = [states, actions, log_probs, returns, advantages]

states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), 
zip(*processed_rollout))
advantages = (advantages - advantages.mean()) / advantages.std()
advantages = Variable(advantages)
returns = Variable(returns)
'''


def mean_std_groups(x, y, group_size):
  '''

:param x:
:type x:
:param y:
:type y:
:param group_size:
:type group_size:
:return:
:rtype:
'''
  num_groups = int(len(x) / group_size)

  x, x_tail = x[:group_size * num_groups], x[group_size * num_groups:]
  x = x.reshape((num_groups, group_size))

  y, y_tail = y[:group_size * num_groups], y[group_size * num_groups:]
  y = y.reshape((num_groups, group_size))

  x_means = x.mean(axis=1)
  x_stds = x.std(axis=1)

  if len(x_tail) > 0:
    x_means = np.concatenate([x_means, x_tail.mean(axis=0, keepdims=True)])
    x_stds = np.concatenate([x_stds, x_tail.std(axis=0, keepdims=True)])

  y_means = y.mean(axis=1)
  y_stds = y.std(axis=1)

  if len(y_tail) > 0:
    y_means = np.concatenate([y_means, y_tail.mean(axis=0, keepdims=True)])
    y_stds = np.concatenate([y_stds, y_tail.std(axis=0, keepdims=True)])

  return x_means, x_stds, y_means, y_stds


def compute_returns(next_value, rewards, masks, discount_factor=0.99):
  R = next_value
  returns = []
  for step in reversed(range(len(rewards))):
    R = rewards[step] + discount_factor * R * masks[step]
    returns.insert(0, R)
  return returns

