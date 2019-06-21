#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
from abc import abstractmethod
from typing import Any

import draugr
from agent.architectures import Architecture
from agent.interfaces.partials.agents.torch_agents.torch_agent import TorchAgent
from agent.interfaces.specifications import GDCS
from neodroid.environments.environment import Environment

__author__ = 'cnheider'

import torch

import agent.utilities as U


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
    self._distribution_parameter_regressor: Architecture = None

    self._deterministic = True

    super().__init__(*args, **kwargs)

  def _build(self, env, verbose=False, **kwargs):

    with draugr.TensorBoardXWriter()as writer:
      dummy_in = torch.rand(1, *self.input_shape)

      model = copy.deepcopy(self._distribution_parameter_regressor)
      model.to('cpu')
      writer.add_graph(model, dummy_in)

    if verbose:
      num_trainable_params = sum(p.numel()
                                 for p in self._distribution_parameter_regressor.parameters()
                                 if p.requires_grad)
      num_params = sum(param.numel() for param in self._distribution_parameter_regressor.parameters())

      draugr.sprint(f'trainable/num_params: {num_trainable_params}/{num_params}\n',
                    highlight=True,
                    color='cyan')

  # endregion

  # region Public

  @property
  def models(self):
    return (self._distribution_parameter_regressor,)

  def save(self, C):
    U.save_model(self._distribution_parameter_regressor, **C)

  def load(self, model_file, evaluation=False):
    print(f'Loading model: {model_file}')
    self._distribution_parameter_regressor = self._policy_arch_spec.constructor(
        **self._policy_arch_spec.kwargs)
    self._distribution_parameter_regressor.load_state_dict(torch.load(model_file))
    if evaluation:
      self._distribution_parameter_regressor = self._distribution_parameter_regressor.eval()
      self._distribution_parameter_regressor.train(False)
    if self._use_cuda:
      self._distribution_parameter_regressor = self._distribution_parameter_regressor.cuda()

  # endregion

  # region Protected

  def _post_io_inference(self, env: Environment):
    self._policy_arch_spec.kwargs['input_shape'] = self._input_shape
    self._policy_arch_spec.kwargs['output_shape'] = self._output_shape
    if hasattr(env.action_space,'is_discrete'):
      self._policy_arch_spec.kwargs['discrete'] = env.action_space.is_discrete
    else:
      self._policy_arch_spec.kwargs['discrete'] = True

      # endregion

  # region Abstract

  @abstractmethod
  def _sample_model(self, state, **kwargs) -> Any:
    raise NotImplementedError

  # endregion
