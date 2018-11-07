#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

import math

import numpy as np
import torch


def entropy(prob):
  return -torch.sum(prob * torch.log(prob), 1)


def log_entropy(log_prob):
  return -torch.sum(torch.exp(log_prob) * log_prob, 1)


def kl_log_probs(log_prob1, log_prob2):
  return -torch.sum(torch.exp(log_prob1) * (log_prob2 - log_prob1), 1)


def kl_probs(prob1, prob2):
  return -torch.sum(prob1 * torch.log(prob2 / prob1), 1)


def to_tensor(obj, dtype=torch.float, device='cpu'):
  if not torch.is_tensor(obj):
    return torch.tensor(obj, device=device, dtype=dtype)
  else:
    return obj.type(dtype).to(device)


def pi_torch(device='cpu'):
  return to_tensor([np.math.pi], device=device)


def normal(x, mu, sigma_sq, device='cpu'):
  a = (-1 * (to_tensor(x, device=device) - mu).pow(2) / (2 * sigma_sq)).exp()
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

import torch
from torch.autograd import Function, Variable

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union + self.inter) \
                         / self.union * self.union
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)



def identity(x):
  return x



def _discount_reward(self, signals, value):
  discounted_r = np.zeros_like(signals)
  running_add = value
  for t in reversed(range(0, len(signals))):
    running_add = running_add * self.gamma + signals[t]
    discounted_r[t] = running_add
  return discounted_r


# choose an action based on state with random noise added for exploration in training
def exploration_action(self, state):
  softmax_action = self._sample_model(state)
  epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
      -1. * self._step_i / self.epsilon_decay
      )
  if np.random.rand() < epsilon:
    action = np.random.choice(self.action_dim)
  else:
    action = np.argmax(softmax_action)
  return action