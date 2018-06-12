#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'



import numpy as np
import torch

import utilities as U





def generalised_advantage_estimate(
    n_step_summary, discount_factor=0.99, tau=0.95, device='cpu'
    ):
  '''
compute GAE(lambda) advantages and discounted returns

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

  signals = U.to_tensor(n_step_summary.signal, device=device, dtype=torch.float)

  non_terminals = U.to_tensor(n_step_summary.non_terminal, device=device, dtype=torch.float)

  value_estimates = U.to_tensor(n_step_summary.value_estimate, device=device, dtype=torch.float)

  T = signals.size()
  T = T[0]
  num_workers = 1
  # T,num_workers,_  = signals.size()

  advs = torch.zeros(T, num_workers, 1).to(device)
  advantage_now = torch.zeros(num_workers, 1).to(device)

  for t in reversed(range(T - 1)):
    signal_now = signals[t]
    value_future = value_estimates[t + 1]
    value_now = value_estimates[t]
    non_terminal_now = non_terminals[t]

    td_error = signal_now + value_future * discount_factor * non_terminal_now - value_now

    advantage_now = advantage_now * discount_factor * tau * non_terminal_now + td_error

    advs[t] = advantage_now

  advantages = advs.squeeze()

  return advantages


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
