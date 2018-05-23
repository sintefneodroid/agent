#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'

from .generalised_advantage_estimation import *


def entropy(probs):
  return -(torch.log(probs) * probs).sum(-1)


def identity(x):
  return x


def kl_log_probs(log_p1, log_p2):
  return -torch.sum(torch.exp(log_p1) * (log_p2 - log_p1), 1)


def _discount_reward(self, signals, value):
  discounted_r = np.zeros_like(signals)
  running_add = value
  for t in reversed(range(0, len(signals))):
    running_add = running_add * self.gamma + signals[t]
    discounted_r[t] = running_add
  return discounted_r


# choose an action based on state with random noise added for exploration in training
def exploration_action(self, state):
  softmax_action = self.__sample_model__(state)
  epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
      -1. * self._step_i / self.epsilon_decay
      )
  if np.random.rand() < epsilon:
    action = np.random.choice(self.action_dim)
  else:
    action = np.argmax(softmax_action)
  return action
