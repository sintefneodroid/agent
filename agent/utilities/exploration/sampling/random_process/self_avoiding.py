#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from agent.utilities.exploration.sampling.random_process import RandomProcess

__author__ = 'cnheider'

import random

import numpy as np


class SelfAvoiding(RandomProcess):

  def __init__(self, num_of_options=4, n=10):
    self.num_of_options = num_of_options
    self.n = n
    self.reset()

  def sample(self, steps=1):
    while (
        (self.x > 0)
        and (self.x < self.n - 1)
        and (self.y > 0)
        and (self.y < self.n - 1)
    ):
      self.a[self.x][self.y] = 1
      if (
          self.a[self.x - 1][self.y]
          and self.a[self.x + 1][self.y]
          and self.a[self.x][self.y - 1]
          and self.a[self.x][self.y + 1]
      ):
        self.deadEnds += 1
        return self.a[self.x - 1][self.y]
      r = random.randrange(1, self.num_of_options + 1)
      if (r == 1) and (not self.a[self.x + 1][self.y]):
        self.x += 1
      elif (r == 2) and (not self.a[self.x - 1][self.y]):
        self.x -= 1
      elif (r == 3) and (not self.a[self.x][self.y + 1]):
        self.y += 1
      elif (r == 4) and (not self.a[self.x][self.y - 1]):
        self.y -= 1

    return self.a[self.x - 1][self.y]

  def reset(self):
    self.deadEnds = 0

    self.a = np.zeros((self.n, self.n))

    self.x = self.n // 2
    self.y = self.n // 2


def main(n=5, trials=3):
  r = SelfAvoiding()

  for t in range(trials):
    print(r.sample())


if __name__ == '__main__':
  main()
