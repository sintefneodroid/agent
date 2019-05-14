#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Any

import draugr
from agent.utilities.specifications.generalised_delayed_construction_specification import GDCS
from agent.architectures import Architecture
from warg import NamedOrderedDictionary

__author__ = 'cnheider'

import torch

import agent.utilities as U
from agent.agents.abstract.torch_agent import TorchAgent


class PolicyAgent(TorchAgent):
  '''
  All policy iteration agents should inherit from this class
  '''

  # region Private

  def __init__(self, *args, **kwargs):
    '''


    :param args:
    :param kwargs:
    '''
    self._policy_arch_spec: GDCS = None
    self._policy_model: Architecture = None

    self._deterministic = True

    super().__init__(*args, **kwargs)

  def build(self, env, **kwargs):
    super().build(env, **kwargs)
    '''
    with tx.SummaryWriter(str(self._base_log_dir))as writer:
      dummy_in = torch.rand(1, *self._observation_space.shape)

      model = copy.deepcopy(self._policy_model)
      model.to('cpu')
      writer.add_graph(model, dummy_in, verbose=self._verbose)
    '''

    num_trainable_params = sum(
        p.numel() for p in self._policy_model.parameters() if p.requires_grad)
    num_params = sum(param.numel() for param in self._policy_model.parameters())
    draugr.sprint(f'trainable/num_params: {num_trainable_params}/{num_params}\n', highlight=True,
                  color='cyan')

  # endregion

  # region Public

  @property
  def models(self):
    return (self._policy_model,)

  def save(self, C):
    U.save_model(self._policy_model, C)

  def load(self, model_file, evaluation=False):
    print(f'Loading model: {model_file}')
    self._policy_model = self._policy_arch_spec.constructor(**self._policy_arch_spec.kwargs)
    self._policy_model.load_state_dict(torch.load(model_file))
    if evaluation:
      self._policy_model = self._policy_model.eval()
      self._policy_model.train(False)
    if self._use_cuda:
      self._policy_model = self._policy_model.cuda()

  # endregion

  # region Protected

  def _maybe_infer_input_output_sizes(self, env, **kwargs):
    super()._maybe_infer_input_output_sizes(env)

    self._policy_arch_spec.kwargs['input_size'] = self._input_size
    self._policy_arch_spec.kwargs['output_size'] = self._output_size

  def _maybe_infer_hidden_layers(self, **kwargs):
    super()._maybe_infer_hidden_layers()

    self._policy_arch_spec.kwargs['hidden_layers'] = self._hidden_layers

  def _train_procedure(self, *args, **kwargs) -> NamedOrderedDictionary:
    return self.train_episodically(*args, **kwargs)

  # endregion

  # region Abstract

  @abstractmethod
  def _sample_model(self, state, *args, **kwargs) -> Any:
    raise NotImplementedError

  @abstractmethod
  def train_episodically(self, rollout, *args, **kwargs) -> U.TR:
    raise NotImplementedError

  # endregion
