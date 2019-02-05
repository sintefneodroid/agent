#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Any

from warg import NamedOrderedDictionary

__author__ = 'cnheider'

import torch

import utilities as U
from agents.abstract.torch_agent import TorchAgent


class PolicyAgent(TorchAgent):
  '''
All policy iteration agents should inherit from this class
'''

  # region Private

  def __init__(self, *args, **kwargs):
    self._policy_arch = None
    self._policy_arch_params = None
    self._policy = None

    self._deterministic = True

    super().__init__(*args, **kwargs)



  # endregion

  # region Public

  def save(self, C):
    U.save_model(self._policy, C)

  def load(self, model_file, evaluation=False):
    print(f'Loading model: { model_file}')
    self._policy = self._policy_arch(**self._policy_arch_params)
    self._policy.load_state_dict(torch.load(model_file))
    if evaluation:
      self._policy = self._policy.eval()
      self._policy.train(False)
    if self._use_cuda:
      self._policy = self._policy.cuda()

  # endregion

  # region Protected

  def _maybe_infer_input_output_sizes(self, env, **kwargs):
    super()._maybe_infer_input_output_sizes(env)

    self._policy_arch_params['input_size'] = self._input_size
    self._policy_arch_params['output_size'] = self._output_size

  def _maybe_infer_hidden_layers(self, **kwargs):
    super()._maybe_infer_hidden_layers()

    self._policy_arch_params['hidden_layers'] = self._hidden_layers

  def _train(self, *args, **kwargs)->NamedOrderedDictionary:
    return self.train_episodically(*args, **kwargs)

  # endregion

  # region Abstract

  @abstractmethod
  def _sample_model(self, state, *args, **kwargs) -> Any:
    raise NotImplementedError

  @abstractmethod
  def train_episodically(self, *args, **kwargs) -> NamedOrderedDictionary:
    raise NotImplementedError

  # endregion
