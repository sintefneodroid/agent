#!/usr/bin/env python3
# coding=utf-8

from utilities.torch_utilities import to_var

__author__ = 'cnheider'

import random

import numpy as np
import torch


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def generalised_advantage_estimate(
    n_step_summary, discount_factor=0.99, gae_tau=0.95, use_cuda=False
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
:param gae_tau:
:type gae_tau:
:return:
:rtype:
'''

  signals = to_var(n_step_summary.signal, use_cuda=use_cuda).view(-1, 1)
  non_terminals = to_var(
      n_step_summary.non_terminal, dtype='float', use_cuda=use_cuda
      ).view(
      -1, 1
      )
  value_estimates = to_var(
      n_step_summary.value_estimate, dtype='float', use_cuda=use_cuda
      ).view(
      -1, 1
      )

  T = len(signals)
  advantage = torch.zeros(1, 1)
  if use_cuda:
    advantage.cuda()
  discounted_return = signals[-1].data[0]
  output = torch.zeros(T, 1)
  output2 = torch.zeros(T, 1)

  for t in reversed(range(T - 1)):
    value_future = value_estimates[t + 1].data
    value_now = value_estimates[t].data
    signal = signals[t].data
    non_terminal = non_terminals[t].data

    discounted_return = signal + discount_factor * non_terminal * discounted_return

    td_error = signal + value_future * discount_factor * non_terminal - value_now
    advantage = advantage * discount_factor * gae_tau * non_terminal + td_error

    output[t] = advantage
    output2[t] = discounted_return

  advantages = torch.cat(output, dim=0)
  discounted_returns = torch.cat(output, dim=0)
  advantages = (advantages - advantages.mean()) / advantages.std()

  advantages = to_var(advantages, use_cuda=use_cuda).view(-1, 1)
  discounted_returns = to_var(discounted_returns, use_cuda=use_cuda).view(-1, 1)

  return advantages, discounted_returns


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


def set_lr(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
