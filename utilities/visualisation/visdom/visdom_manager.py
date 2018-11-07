#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
__author__ = 'cnheider'


class VisdomManager:

  def __init__(self):
    self.VISDOM_PROCESS = None

  def start_visdom_process(self, configuration):
    if configuration.START_VISDOM_SERVER:
      print('Starting visdom server process')
      import subprocess

      self.VISDOM_PROCESS = subprocess.Popen(
          ['python3', 'draugr_utilities/visualisation/run_visdom_server.py'],
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          )

  def stop_visdom_process(self):
    if self.VISDOM_PROCESS:
      input(
          'Keeping visdom running, pressing '
          'enter will terminate visdom process..'
          )
      self.VISDOM_PROCESS.terminate()
