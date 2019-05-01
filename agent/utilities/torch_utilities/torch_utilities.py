#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import random

import numpy
import torch

__author__ = 'cnheider'


def seed(s):
  random.seed(s)
  numpy.random.seed(s)
  torch.manual_seed(s)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(s)


def add_indent(s_, numSpaces):
  s = s_.split('\n')
  if len(s) == 1:  # don't do anything for single-line stuff
    return s_
  first = s.pop(0)
  s = [(numSpaces * ' ') + line for line in s]
  s = '\n'.join(s)
  s = first + '\n' + s
  return s


def entropy(prob):
  return -torch.sum(prob * torch.log(prob), 1)


def log_entropy(log_prob):
  return -torch.sum(torch.exp(log_prob) * log_prob, 1)


def to_tensor(obj, dtype=torch.float, device='cpu'):
  if not torch.is_tensor(obj):
    if isinstance(obj, numpy.ndarray):
      return torch.from_numpy(numpy.array(obj)).to(device=device, dtype=dtype)
    return torch.tensor(obj, device=device, dtype=dtype)
  else:
    return obj.type(dtype).to(device)


def pi_torch(device='cpu'):
  return to_tensor([numpy.math.pi], device=device)


'''
def normal(x, mean, sigma_sq, device='cpu'):
  a = (-1 * (to_tensor(x, device=device) - mean).pow(2) / (2 * sigma_sq)).exp()
  b = 1 / (2 * sigma_sq * pi_torch(device=device).expand_as(sigma_sq)).sqrt()
  return a * b
'''


def normal_entropy(std):
  var = std.pow(2)
  ent = 0.5 + 0.5 * torch.log(2 * var * math.pi)
  return ent.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
  var = std.pow(2)
  log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
  return log_density.sum(1, keepdim=True)


import torch


def identity(x):
  return x


def _discount_reward(self, signals, value):
  discounted_r = numpy.zeros_like(signals)
  running_add = value
  for t in reversed(range(0, len(signals))):
    running_add = running_add * self.gamma + signals[t]
    discounted_r[t] = running_add
  return discounted_r


# choose an action based on state with random noise added for exploration in training
def exploration_action(self, state):
  softmax_action = self._sample_model(state)
  epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * numpy.exp(
      -1. * self._step_i / self.epsilon_decay
      )
  if numpy.random.rand() < epsilon:
    action = numpy.random.choice(self.action_dim)
  else:
    action = numpy.argmax(softmax_action)
  return action


def reverse_channel_transform(inp):
  inp = inp.transpose((1, 2, 0))
  inp = inp * 255.0
  inp = numpy.clip(inp, 0, 255).astype(numpy.uint8)
  return inp


def channel_transform(inp):
  inp = inp / 255.0
  inp = numpy.clip(inp, 0, 1)
  inp = inp.transpose((2, 0, 1))
  return inp
