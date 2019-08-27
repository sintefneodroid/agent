#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
from abc import abstractmethod
from typing import Any

from draugr.visualisation import sprint
from draugr.writers import TensorBoardPytorchWriter, MockWriter
from neodroidagent.architectures import Architecture
from neodroidagent.architectures.distributional.categorical import CategoricalMLP
from neodroidagent.architectures.distributional.normal import MultiDimensionalNormalMLP
from neodroidagent.interfaces.torch_agent import TorchAgent
from draugr.writers.writer import Writer
from neodroid.environments.environment import Environment
from warg.gdkc import GDKC

__author__ = 'cnheider'

import torch


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
    self._policy_arch_spec: GDKC = None
    self._distribution_regressor: Architecture = None

    self._optimiser_spec = None

    self._deterministic = True

    super().__init__(*args, **kwargs)

  def __build__(self,
                env: Environment, metric_writer: Writer = MockWriter(),
                print_model_repr=True,
                **kwargs):

    self._distribution_regressor = self._policy_arch_spec().to(
        self._device)

    self.optimiser = self._optimiser_spec(self._distribution_regressor.parameters())

    if metric_writer:
      dummy_in = torch.rand(1, *self.input_shape)

      model = copy.deepcopy(self._distribution_regressor)
      model.to('cpu')

      if isinstance(metric_writer, TensorBoardPytorchWriter):
        metric_writer.graph(model, dummy_in)

    if print_model_repr:
      sprint(f'Distribution regressor: {self._distribution_regressor}',
                    highlight=True,
                    color='cyan')

  # endregion

  # region Public

  @property
  def models(self)-> dict:
    return {'_distribution_regressor':self._distribution_regressor,}

  # endregion

  # region Protected

  def _post_io_inference(self, env: Environment):

    self._policy_arch_spec.kwargs['input_shape'] = self._input_shape
    if env.action_space.is_discrete:
      self._policy_arch_spec = GDKC(CategoricalMLP, self._policy_arch_spec.kwargs)
      self._policy_arch_spec.kwargs['output_shape'] = self._output_shape
    else:
      self._policy_arch_spec = GDKC(MultiDimensionalNormalMLP, self._policy_arch_spec.kwargs)
      self._policy_arch_spec.kwargs['output_shape'] = self._output_shape

      # endregion

  # region Abstract

  @abstractmethod
  def _sample_model(self, state, **kwargs) -> Any:
    raise NotImplementedError

  # endregion
