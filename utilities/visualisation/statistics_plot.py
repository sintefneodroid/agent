#!/usr/bin/env python3
# coding=utf-8
import matplotlib
import torch

__author__ = 'cnheider'

import csv

import pylab as plt

import utilities as U

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
  from IPython import display

plt.ion()

_file_name_1 = '/home/heider/Downloads/CurriculumTraining/without/Neodroid' \
               '-configs_curriculum_curriculum_config-1803061120.entropys.csv'
_file_name_2 = '/home/heider/Downloads/CurriculumTraining/with/Neodroid' \
               '-configs_curriculum_curriculum_config-1803060209.entropys.csv'


# _file_name = '/home/heider/Github/Neodroid/agent/logs/Neodroid-configs_curriculum_config-1801231302
# .episode_durations.csv'


def ma_plot(file_name, name):
  with open(file_name, 'r') as f:
    agg = U.Aggregator()
    agg_ma = U.Aggregator()

    reader = csv.reader(f, delimiter=' ', quotechar='|')
    for line in reader:
      if line and line[0] != '':
        agg.append(float(line[0][1:-2]))
        ma = agg.moving_average()
        agg_ma.append(ma)

    plt.plot(agg_ma.values, name)


def simple_plot(file_name):
  with open(file_name, 'r') as f:
    agg = U.Aggregator()

    reader = csv.reader(f, delimiter=' ', quotechar='|')
    for line in reader:
      agg.append(float(line[0]))

    plt.plot(agg.values)
    plt.show()


def plot_durations(episode_durations):
  plt.figure(2)
  plt.clf()
  durations_t = torch.tensor(episode_durations, dtype=torch.float)
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
  ma_plot(_file_name_1, 'NoCur')
  ma_plot(_file_name_2, 'Cur')
  # simple_plot(_file_name_1)
  # simple_plot(_file_name_2)
  plt.show()
