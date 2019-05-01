#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'

import csv

import matplotlib.pyplot as plt

from agent import utilities as U

# print(plt.style.available)
plot_style = 'fivethirtyeight'
# plot_style='bmh'
# plot_style='ggplot'
plt.style.use('seaborn-poster')
plt.style.use(plot_style)
plt.rcParams['axes.edgecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['patch.edgecolor'] = '#ffffff'
plt.rcParams['patch.facecolor'] = '#ffffff'
plt.rcParams['savefig.edgecolor'] = '#ffffff'
plt.rcParams['savefig.facecolor'] = '#ffffff'
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# set up matplotlib
is_ipython = 'inline' in plt.get_backend()
if is_ipython:
  pass

plt.ion()


def simple_plot(file_path, name='Statistic Name'):
  with open(file_path, 'r') as f:
    agg = U.StatisticAggregator()

    reader = csv.reader(f, delimiter=' ', quotechar='|')
    for line in reader:
      agg.append(float(line[0]))

    plt.plot(agg.values)
    plt.title(name)

    plt.show()


if __name__ == '__main__':
  #  import configs.base_config as C

  # _list_of_files = list(C.LOG_DIRECTORY.glob('*.csv'))
  # _latest_model = max(_list_of_files, key=os.path.getctime)

  from tkinter import Tk
  from tkinter.filedialog import askopenfilename

  # import easygui
  # print easygui.fileopenbox()

  Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
  file_path = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
  file_name = file_path.split('/')[-1]
  simple_plot(file_path, file_name)
