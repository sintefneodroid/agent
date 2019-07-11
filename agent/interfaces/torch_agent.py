#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from agent.interfaces.agent import Agent

tqdm.monitor_interval = 0

__author__ = 'cnheider'


class TorchAgent(Agent, ABC):
  '''
All agent should inherit from this class
'''

  # region Private

  def __init__(self,
               *args,
               use_cuda=False,
               cuda_device_id=0,
               **kwargs):
    self._use_cuda = use_cuda
    self._device = torch.device(f'cuda:{cuda_device_id}'
                                if torch.cuda.is_available() and self._use_cuda else 'cpu')

    super().__init__(*args, **kwargs)

  @property
  def device(self) -> torch.device:
    return self._device

  # endregion


  # region Static

  @staticmethod
  def _update_target(*, target_model, source_model, target_update_tau=3e-3):
    assert 0.0 <= target_update_tau <= 1.0
    for target_param, param in zip(target_model.parameters(), source_model.parameters()):
      target_param.data.copy_(target_update_tau * param.data
                              + (1 - target_update_tau) * target_param.data
                              )

  # endregion

