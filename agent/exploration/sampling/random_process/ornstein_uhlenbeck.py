#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
import numpy as np

from .annealed_guassian import AnnealedGaussianProcess


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):

  def __init__(self,
               theta,
               mean=0.,
               sigma=1.,
               dt=1e-2,
               x_0=None,
               size=1,
               sigma_min=None,
               n_steps_annealing=1000,
               ):
    super().__init__(mean=mean,
                     sigma=sigma,
                     sigma_min=sigma_min,
                     n_steps_annealing=n_steps_annealing
                     )
    self.theta = theta
    self.mean = mean
    self.dt = dt
    self.x_0 = x_0
    self.size = size
    self.reset()

  def sample(self):
    x = (self.x_prev + self.theta * (self.mean - self.x_prev)
         * self.dt + self.current_sigma * np.sqrt(self.dt) *
         np.random.normal(size=self.size))
    self.x_prev = x
    self.n_steps += 1
    return x

  def reset(self):
    super().reset()
    self.x_prev = self.x_0 if self.x_0 is not None else np.zeros(self.size)


def main():
  random_process = OrnsteinUhlenbeckProcess(0.5)

  for i in range(1000):
    print(random_process.sample())


if __name__ == '__main__':
  main()
