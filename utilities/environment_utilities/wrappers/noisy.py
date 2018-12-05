#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

import gym
import numpy as np
from gym.spaces.box import Box


class NoisyWrapper(gym.ObservationWrapper):
  """Make observation dynamic by adding noise"""

  def __init__(self, env=None, percent_pad=5, bottom_margin=20):
    '''
    # doom 20px bottom is useless

    :param env:
    :param percent_pad:
    :param bottom_margin:
    '''
    super().__init__(env)
    self.original_shape = env.observation_space.shape
    new_side = int(round(max(self.original_shape[:-1]) * 100. / (100. - percent_pad)))
    self.new_shape = [new_side, new_side, 3]
    self.observation_space = Box(0.0, 255.0, self.new_shape)
    self.bottom_margin = bottom_margin
    self.ob = None

  def _observation(self, obs):
    im_noise = np.random.randint(0, 256, self.new_shape).astype(obs.dtype)
    im_noise[:self.original_shape[0] - self.bottom_margin, :self.original_shape[1], :] = obs[
                                                                                         :-self.bottom_margin,
                                                                                         :, :]
    self.ob = im_noise
    return im_noise

  # def render(self, mode='human', close=False):
  #     temp = self.env.render(mode, close)
  #     return self.ob
