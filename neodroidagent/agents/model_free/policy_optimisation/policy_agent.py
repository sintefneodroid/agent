#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
from abc import abstractmethod
from typing import Any

import draugr
from neodroidagent.architectures import Architecture
from neodroidagent.architectures.distributional.categorical import CategoricalMLP
from neodroidagent.architectures.distributional.normal import MultiDimensionalNormalMLP
from neodroidagent.interfaces.specifications import GDCS
from neodroidagent.interfaces.torch_agent import TorchAgent
from draugr.writers.writer import Writer
from neodroid.environments.environment import Environment

__author__ = 'cnheider'

import torch

import neodroidagent.utilities as U


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
    self._distribution_regressor: Architecture = None

    self._deterministic = True

    super().__init__(*args, **kwargs)

  def _build(self, env: Environment, stat_writer: Writer = None, print_model_repr=True, **kwargs):

    if stat_writer:
      dummy_in = torch.rand(1, *self.input_shape)

      model = copy.deepcopy(self._distribution_regressor)
      model.to('cpu')

      if isinstance(stat_writer, draugr.TensorBoardXWriter):
        stat_writer._graph(model, dummy_in)

    if print_model_repr:
      draugr.sprint(f'Distribution regressor: {self._distribution_regressor}',
                    highlight=True,
                    color='cyan')

  # endregion

  # region Public

  @property
  def models(self):
    return (self._distribution_regressor,)

  def save(self, C):
    U.save_model(self._distribution_regressor, **C)

  def load(self, model_file, evaluation=False):
    print(f'Loading model: {model_file}')
    self._distribution_regressor = self._policy_arch_spec.constructor(
        **self._policy_arch_spec.kwargs)
    self._distribution_regressor.load_state_dict(torch.load(model_file))
    if evaluation:
      self._distribution_regressor = self._distribution_regressor.eval()
      self._distribution_regressor.train(False)
    if self._use_cuda:
      self._distribution_regressor = self._distribution_regressor.cuda()

  # endregion

  # region Protected

  def _post_io_inference(self, env: Environment):

    self._policy_arch_spec.kwargs['input_shape'] = self._input_shape
    if env.action_space.is_discrete:
      self._policy_arch_spec = GDCS(CategoricalMLP, self._policy_arch_spec.kwargs)
      self._policy_arch_spec.kwargs['output_shape'] = self._output_shape
    else:
      self._policy_arch_spec = GDCS(MultiDimensionalNormalMLP, self._policy_arch_spec.kwargs)
      self._policy_arch_spec.kwargs['output_shape'] = self._output_shape

      # endregion

  # region Abstract

  @abstractmethod
  def _sample_model(self, state, **kwargs) -> Any:
    raise NotImplementedError

  # endregion
