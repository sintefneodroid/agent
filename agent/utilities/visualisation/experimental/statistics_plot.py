#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import matplotlib
import numpy as np
import torch

__author__ = 'cnheider'

import csv

import matplotlib.pyplot as plt

from agent import utilities as U

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
  from IPython import display

plt.ion()


def ma_plot(file_name, name):
  with open(file_name, 'r') as f:
    agg = U.StatisticAggregator()
    agg_ma = U.StatisticAggregator()

    reader = csv.reader(f, delimiter=' ', quotechar='|')
    for line in reader:
      if line and line[0] != '':
        agg.append(float(line[0][1:-2]))
        ma = agg.calc_moving_average()
        agg_ma.append(ma)

    plt.plot(agg_ma.values)
    plt.title(name)


def simple_plot(file_name, name='Statistic Name'):
  with open(file_name, 'r') as f:
    agg = U.StatisticAggregator()

    reader = csv.reader(f, delimiter=' ', quotechar='|')
    for line in reader:
      agg.append(float(line[0]))

    # plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),            arrowprops=dict(facecolor='black',
    # shrink=0.05),)

    plt.plot(agg.values)
    plt.title(name)


def error_plot(results, interval=1, file_name=''):
  # if results is not np.ndarray:
  # results = np.ndarray(results)

  y = np.mean(results, axis=0)
  error = np.std(results, axis=0)

  x = range(0, results.shape[1] * interval, interval)
  fig, ax = plt.subplots(1, 1, figsize=(6, 5))
  plt.xlabel('Time step')
  plt.ylabel('Average Reward')
  ax.errorbar(x, y, yerr=error, fmt='-o')
  # plt.savefig(file_name + '.png')


def plot_durations(episode_durations):
  plt.figure(2)
  plt.clf()
  durations_t = U.to_tensor(episode_durations)
  plt.title('Training...')
  plt.xlabel('Episode')
  plt.ylabel('Duration')
  plt.plot(durations_t.numpy())
  # Take 100 episode averages and plot them too
  if len(durations_t) >= 100:
    means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(99), means))
    plt.plot(means.numpy())

  plt.pause(0.001)  # pause a bit so that plots are updated
  if is_ipython:
    display.clear_output(wait=True)
    display.display(plt.gcf())


if __name__ == '__main__':

  import agent.configs.base_config as C

  _list_of_files = list(C.LOG_DIRECTORY.glob('*.csv'))
  _latest_model = max(_list_of_files, key=os.path.getctime)

  # ma_plot(_file_name_1, 'NoCur')
  # ma_plot(_file_name_2, 'Cur')
  # simple_plot(_latest_model)
  a = [0, 92, 3, 2, 5, 644, 34, 36, 423, 421]
  b = [215, 92, 6, 1, 5, 644, 328, 32, 413, 221]
  c = [62, 68, 8, 25, 7, 611, 29, 38, 421, 425]
  d = np.array(zip([a, b, c]))
  error_plot(d)

  plt.show()
