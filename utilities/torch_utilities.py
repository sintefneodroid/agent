#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'

import math
from collections import namedtuple

import numpy as np
import torch


class Variable(torch.autograd.Variable):

  def __init__(self, data, *args, use_cuda=True, **kwargs):
    if use_cuda:
      data = data.cuda()
    super().__init__(data, *args, **kwargs)


def cuda_if(torch_object, cuda):
  return torch_object.cuda() if cuda else torch_object


def entropy(prob):
  return -torch.sum(prob * torch.log(prob), 1)


def log_entropy(log_prob):
  return -torch.sum(torch.exp(log_prob) * log_prob, 1)


def kl_log_probs(log_prob1, log_prob2):
  return -torch.sum(torch.exp(log_prob1) * (log_prob2 - log_prob1), 1)


def kl_probs(prob1, prob2):
  return -torch.sum(prob1 * torch.log(prob2 / prob1), 1)


# noinspection PyCallingNonCallable
def to_tensor_device(x, dtype=torch.float, device='cpu'):
  if type(x) is not torch.Tensor:
    return torch.tensor(x, device=device, dtype=dtype)
  else:
    return x.type(dtype).to(device)


def pi_torch(device='cpu'):
  to_tensor_device([np.math.pi], device=device)


def normal(x, mu, sigma_sq, device='cpu'):
  a = (-1 * (Variable(x) - mu).pow(2) / (2 * sigma_sq)).exp()
  b = 1 / (2 * sigma_sq * pi_torch(device=device).expand_as(sigma_sq)).sqrt()
  return a * b


def normal_entropy(std):
  var = std.pow(2)
  ent = 0.5 + 0.5 * torch.log(2 * var * math.pi)
  return ent.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
  var = std.pow(2)
  log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
  return log_density.sum(1, keepdim=True)


OptimiserSpecification = namedtuple('OptimiserSpecification', ['constructor', 'kwargs'])
