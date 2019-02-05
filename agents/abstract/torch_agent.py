#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from agents.abstract.agent import Agent
from tqdm import tqdm

tqdm.monitor_interval = 0

__author__ = 'cnheider'


class TorchAgent(Agent):
  '''
All agent should inherit from this class
'''

  # region Private

  def __init__(self, use_cuda=False, *args, **kwargs):
    self._hidden_layers = None
    self._use_cuda = use_cuda
    self._device = torch.device('cuda:0' if torch.cuda.is_available() and self._use_cuda else 'cpu')

    super().__init__(*args, **kwargs)

  @property
  def device(self):
    return self._device
