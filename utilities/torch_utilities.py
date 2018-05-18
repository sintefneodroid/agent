#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'

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


def entropy(p):
  return -torch.sum(p * torch.log(p), 1)


def log_entropy(lp):
  return -torch.sum(torch.exp(lp) * lp, 1)


def kl_log_probs(log_p1, log_p2):
  return -torch.sum(torch.exp(log_p1) * (log_p2 - log_p1), 1)


def to_var(x, dtype='float', volatile=False, use_cuda=False, unsqueeze=False):
  if unsqueeze:
    var = torch.autograd.Variable(
        to_tensor(x, dtype=dtype, use_cuda=use_cuda).unsqueeze(0), volatile=volatile
        )
  else:
    var = torch.autograd.Variable(
        to_tensor(x, dtype=dtype, use_cuda=use_cuda), volatile=volatile
        )
  return var


def to_tensor(x, dtype='float', use_cuda=True):
  FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
  LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
  ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
  if dtype == 'long':
    x = np.array(x, dtype=np.long).tolist()
    return LongTensor(x)
  elif dtype == 'byte':
    x = np.array(x, dtype=np.byte).tolist()
    return ByteTensor(x)
  x = np.array(x, dtype=np.float64).tolist()
  return FloatTensor(x)


def pi_torch(use_cuda=False):
  to_var([np.math.pi], use_cuda=use_cuda)


def normal(x, mu, sigma_sq, use_cuda=False):
  a = (-1 * (Variable(x) - mu).pow(2) / (2 * sigma_sq)).exp()
  b = 1 / (2 * sigma_sq * pi_torch(use_cuda).expand_as(sigma_sq)).sqrt()
  return a * b


OSpec = namedtuple('OptimiserSpecification', ['constructor', 'kwargs'])
