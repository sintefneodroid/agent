#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from draugr import save_model, load_latest_model
from neodroidagent.agents.agent import Agent
from neodroidagent.architectures.architecture import Architecture
from warg import passes_kws_to, super_init_pass_on_kws

tqdm.monitor_interval = 0

__author__ = 'Christian Heider Nielsen'


@super_init_pass_on_kws(super_base=Agent)
class TorchAgent(Agent, ABC):
  '''
All agent should inherit from this class
'''

  # region Private

  def __init__(self,
               *,
               device: str = 'cuda',
               **kwargs):
    super().__init__(**kwargs)
    self._device = torch.device(device
                                if torch.cuda.is_available()
                                   and device != 'cpu'
                                else 'cpu')

  @property
  def device(self) -> torch.device:
    return self._device

  # endregion

  # region Public
  @property
  @abstractmethod
  def models(self) -> Dict[str, Architecture]:
    raise NotImplementedError

  @passes_kws_to(save_model)
  def save(self, **kwargs) -> None:
    for k, v in self.models.items():
      save_model(v, model_name=k, **kwargs)

  def load(self,
           model_path: Path,
           evaluation=False,
           **kwargs) -> None:

    if len(self.models.items()) > 1:
      for k, v in self.models.items():
        model_p = model_path / f'-{k}'
        if model_p.exists():
          print('Loading model: ' + str(model_path))
          model = getattr(self, k)
          model.load_state_dict(torch.load(model_p))

          if evaluation:
            model = model.eval()
            model.train(False)

          model = model.to(self._device)

          setattr(self, k, model)
    else:
      for k, v in self.models.items():
        model_p = load_latest_model(model_path)
        if model_p:
          print('Loading model: ' + str(model_path))
          model = getattr(self, k)
          model.load_state_dict(model_p)

          if evaluation:
            model = model.eval()
            model.train(False)

          model = model.to(self._device)

          setattr(self, k, model)

  # endregion

  # region Static

  @staticmethod
  def _update_target(*, target_model, source_model, copy_percentage=3e-2):
    assert 0.0 <= copy_percentage <= 1.0
    for target_param, param in zip(target_model.parameters(), source_model.parameters()):
      target_param.data.copy_(copy_percentage * param.data
                              + (1 - copy_percentage) * target_param.data
                              )

  # endregion
