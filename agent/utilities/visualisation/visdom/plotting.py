#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

import numpy as np
import visdom

vis = visdom.Visdom()


def plot_episode_stats(stats):
  # Plot the mean of last 100 episode rewards over time.
  vis.line(
      X=np.arange(len(stats.signal_mas)),
      Y=np.array(stats.signal_mas),
      win='DDPG MEAN REWARD (100 episodes)',
      opts=dict(
          title=('DDPG MEAN REWARD (100 episodes)'),
          ylabel='MEAN REWARD (100 episodes)',
          xlabel='Episode',
          ),
      )

  # Plot time steps and episode number.
  vis.line(
      X=np.cumsum(stats.episode_lengths),
      Y=np.arange(len(stats.episode_lengths)),
      win='DDPG Episode per time step',
      opts=dict(
          title=('DDPG Episode per time step'), ylabel='Episode', xlabel='Time Steps'
          ),
      )
