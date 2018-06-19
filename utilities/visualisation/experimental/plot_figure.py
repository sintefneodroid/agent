#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

import matplotlib.pyplot as plt
import numpy as np


def plot_figure(episodes, eval_rewards, env_id):
  episodes = np.array(episodes)
  eval_rewards = np.array(eval_rewards)
  np.savetxt(f'./output/{env_id}_ppo_episodes.txt', episodes)
  np.savetxt(f'./output/{env_id}_ppo_eval_rewards.txt', eval_rewards)

  plt.figure()
  plt.plot(episodes, eval_rewards)
  plt.title('%s' % env_id)
  plt.xlabel('Episode')
  plt.ylabel('Average Reward')
  plt.legend(['PPO'])
  plt.savefig(f'./output/{env_id}_ppo.png')
