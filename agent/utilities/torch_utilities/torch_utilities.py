#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import random
from typing import Iterable, Sequence

import numpy
import torch
from torch.utils.data import Dataset

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


def shannon_entropy(prob):
  return - torch.sum(prob * torch.log2(prob), -1)


def log_shannon_entropy(log_prob):
  return - torch.sum(torch.pow(2, log_prob) * log_prob, -1)
  # return - torch.sum(torch.exp(log_prob) * log_prob, -1)


def to_tensor(obj, dtype=torch.float, device='cpu', non_blocking=True):
  if torch.is_tensor(obj):
    return obj.to(device, dtype=dtype, non_blocking=non_blocking)

  if isinstance(obj, numpy.ndarray):
    if torch.is_tensor(obj[0]):
      return torch.cat(obj.tolist())
    return torch.from_numpy(obj).to(device=device,
                                    dtype=dtype,
                                    non_blocking=non_blocking)

  if not isinstance(obj, Sequence):
    obj = [obj]
  elif not isinstance(obj, list) and isinstance(obj, Iterable):
    obj = [*obj]

  if isinstance(obj, list):
    if torch.is_tensor(obj[0]) and len(obj[0].size()) > 0:
      return torch.cat(obj)

  return torch.tensor(obj, device=device, dtype=dtype)


def torch_pi(device='cpu'):
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
  return ent.sum(dim=-1, keepdim=True)


def differential_entropy_gaussian(std):
  return torch.log(std * torch.sqrt(2 * torch_pi())) + .5


def normal_log_density(x, mean, log_std, std):
  var = std.pow(2)
  log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
  return log_density.sum(1, keepdim=True)


def identity(x):
  return x


def discount_signal(self, signals, value):
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


class NonSequentialDataset(Dataset):
  """
   * ``N`` - number of parallel environments
   * ``T`` - number of time steps explored in environments

  Dataset that flattens ``N*T*...`` arrays into ``B*...`` (where ``B`` is equal to ``N*T``) and returns
  such rows
  one by one. So basically we loose information about sequence order and we return
  for example one state, action and reward per row.

  It can be used for ``Model``'s that does not need to keep the order of events like MLP models.

  For ``LSTM`` use another implementation that will slice the dataset differently
  """

  def __init__(self, *arrays: numpy.ndarray) -> None:
    """
    :param arrays: arrays to be flattened from ``N*T*...`` to ``B*...`` and returned in each call to get
    item
    """
    super().__init__()
    self.arrays = [array.reshape(-1, *array.shape[2:]) for array in arrays]

  def __getitem__(self, index):
    return [array[index] for array in self.arrays]

  def __len__(self):
    return len(self.arrays[0])


if __name__ == '__main__':
  eq = to_tensor([0.5, 0.5])
  print(shannon_entropy(eq))
  print(log_shannon_entropy(torch.log2(eq)))
  print(differential_entropy_gaussian(to_tensor(1)))
  print(normal_entropy(to_tensor(1)))
