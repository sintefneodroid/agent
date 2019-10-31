#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
from abc import abstractmethod
from typing import Any

import torch

from draugr.torch_utilities.to_tensor import to_tensor
from draugr.writers import MockWriter, TensorBoardPytorchWriter
from draugr.writers.terminal import sprint
from draugr.writers.writer import Writer
from neodroid.utilities.spaces import ActionSpace, ObservationSpace, SignalSpace
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.architectures import Architecture
from neodroidagent.architectures.distributional.categorical import CategoricalMLP
from neodroidagent.architectures.distributional.normal import MultiDimensionalNormalMLP
from neodroidagent.architectures.mock import MockArchitecture
from neodroidagent.memory import TrajectoryBuffer
from warg.gdkc import GDKC
from warg.kw_passing import super_init_pass_on_kws

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 23/09/2019
           '''


@super_init_pass_on_kws
class PolicyAgent(TorchAgent):
  '''
  All policy iteration agents should inherit from this class
  '''

  # region Private

  def __init__(self,
               evaluation_function=torch.nn.CrossEntropyLoss(),
               trajectory_trace=TrajectoryBuffer(),
               policy_arch_spec=GDKC(CategoricalMLP,
                                     input_shape=None,
                                     # Obtain from environment
                                     hidden_layers=None,
                                     output_shape=None,
                                     # Obtain from environment
                                     ),
               discount_factor=0.95,
               use_batched_updates=False,
               batch_size=5,
               policy_entropy_regularisation=1,

               optimiser_spec=GDKC(torch.optim.Adam,
                                   lr=3e-4,
                                   weight_decay=3e-3,
                                   eps=3e-2),
               state_type=torch.float,
               signals_tensor_type=torch.float,
               discrete=True,

               std=.3,
               distribution_regressor: Architecture = MockArchitecture(),
               deterministic=True,
               **kwargs):
    '''

    :param evaluation_function:
    :param trajectory_trace:
    :param policy_arch_spec:
    :param discount_factor:
    :param use_batched_updates:
    :param batch_size:
    :param policy_entropy_regularisation:
    :param signal_clipping:
    :param signal_clip_high:
    :param signal_clip_low:
    :param optimiser_spec:
    :param state_type:
    :param signals_tensor_type:
    :param discrete:
    :param grad_clip:
    :param grad_clip_low:
    :param grad_clip_high:
    :param std:
    :param distribution_regressor:
    :param deterministic:
    :param kwargs:
    '''
    super().__init__(**kwargs)

    self._accumulated_error = to_tensor(0.0, device=self._device)
    self._evaluation_function = evaluation_function
    self._trajectory_trace = trajectory_trace
    self._policy_arch_spec = policy_arch_spec
    self._discount_factor = discount_factor
    self._use_batched_updates = use_batched_updates
    self._batch_size = batch_size
    self._policy_entropy_regularisation = policy_entropy_regularisation

    self._discrete = discrete

    self._optimiser_spec = optimiser_spec
    self._state_type = state_type
    self._signals_tensor_type = signals_tensor_type

    self._std = std
    self._distribution_regressor = distribution_regressor
    self._deterministic = deterministic

  def __build__(self,
                observation_space: ObservationSpace,
                action_space: ActionSpace,
                signal_space: SignalSpace,
                metric_writer: Writer = MockWriter(),
                print_model_repr=True,
                **kwargs):
    self._policy_arch_spec.kwargs['input_shape'] = self._input_shape
    if action_space.is_discrete:
      self._policy_arch_spec = GDKC(CategoricalMLP, self._policy_arch_spec.kwargs)
      self._policy_arch_spec.kwargs['output_shape'] = self._output_shape
    else:
      self._policy_arch_spec = GDKC(MultiDimensionalNormalMLP, self._policy_arch_spec.kwargs)
      self._policy_arch_spec.kwargs['output_shape'] = self._output_shape

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
  def models(self) -> dict:
    return {'_distribution_regressor':self._distribution_regressor, }

  # endregion

  # region Abstract

  @abstractmethod
  def _sample_model(self, state, **kwargs) -> Any:
    raise NotImplementedError

  # endregion
