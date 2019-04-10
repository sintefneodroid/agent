# !/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

import gym


class SkipEnv(gym.Wrapper):
  """Skip timesteps: repeat action, accumulate signal, take last obs."""

  def __init__(self, env=None, skips=4):
    super().__init__(env)
    self.skip = skips

  def _step(self, action):
    total_signal = 0
    obs = 0
    terminal = False
    info = {}

    for i in range(0, self.skip):
      obs, signal, terminal, info = self.env.act(action)
      total_signal += signal
      info['steps'] = i + 1
      if terminal:
        break

    return obs, total_signal, terminal, info
