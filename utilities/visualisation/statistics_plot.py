#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'

import csv

import pylab as plt

import utilities as U

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


if __name__ == '__main__':
  ma_plot(_file_name_1, 'NoCur')
  ma_plot(_file_name_2, 'Cur')
  # simple_plot(_file_name_1)
  # simple_plot(_file_name_2)
  plt.show()
