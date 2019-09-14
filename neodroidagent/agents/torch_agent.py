#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm
from warg.kw_passing import super_init_pass_on_kws
from draugr.torch_utilities import save_model
from neodroidagent.agents.agent import Agent
from neodroidagent.architectures.architecture import Architecture


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

  def save(self, model_path: Path, **kwargs) -> None:
    for k, v in self.models.items():
      save_model(v, name=k, **kwargs)

  def load(self,
           model_path: Path,
           evaluation=False,
           **kwargs) -> None:

    self.__build__(None, **kwargs)

    for k, v in self.models.items():
      model_p = model_path / f'-{k}'
      print('Loading model: ' + str(model_path))
      model = getattr(self, k)
      model.load_state_dict(torch.load(model_p))

      if evaluation:
        model = model.eval()
        model.train(False)

      model = model.to(self._device)

      setattr(self, k, model)

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
