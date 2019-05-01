#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

import gym


class ClippedRewardEnv(gym.RewardWrapper):
  """Clip signal."""

  def __init__(self,
               env=None,
               negative_clip=0.0,
               positive_clip=None):
    super().__init__(env)
    self._negative_clip = negative_clip
    self._positive_clip = positive_clip

  def _reward(self, signal):
    new_signal = self._negative_clip if self._positive_clip and signal < self._negative_clip else signal
    new_signal = self._positive_clip if self._positive_clip and signal > self._positive_clip else new_signal
    return new_signal
