#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

from .random_process import RandomProcess


class AnnealedGaussianProcess(RandomProcess):

  def sample(self):
    pass

  def __init__(self, mean, sigma, sigma_min, n_steps_annealing):
    self.mean = mean
    self.sigma = sigma
    self.n_steps = 0

    if sigma_min is not None:
      self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
      self.c = sigma
      self.sigma_min = sigma_min
    else:
      self.m = 0.
      self.c = sigma
      self.sigma_min = sigma

  def reset(self):
    self.n_steps = 0

  @property
  def current_sigma(self):
    sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
    return sigma


def main():
  # AnnealedGaussianProcess()
  pass


if __name__ == '__main__':
  main()
