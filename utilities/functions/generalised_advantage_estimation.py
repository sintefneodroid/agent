#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'


import random

import numpy as np
import torch


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def gae(signals, value_estimates, masks, gamma=0.99, glp=0.95):  # , cuda=True):
  T = len(signals)

  advantages = np.zeros(T)

  advantage_t = 0
  for t in reversed(range(T - 1)):
    delta = signals[t] + \
            value_estimates[t + 1].data.numpy()[0] * gamma * masks[t] - \
            value_estimates[t].data.numpy()[0]
    advantage_t = delta + advantage_t * gamma * glp * masks[t]
    advantages[t] = advantage_t

  value_estimates = [value.data.numpy()[0][0] for value in value_estimates]
  discounted_returns = value_estimates + advantages

  return advantages, discounted_returns


def mean_std_groups(x, y, group_size):
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
