#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Any

from warg import NamedOrderedDictionary

__author__ = 'cnheider'

import torch

import utilities as U
from agents.abstract.agent import Agent


class PolicyAgent(Agent):
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

  '''
        is action_space.is_singular:
      if not self._naive_max_policy:
        if action_space.is_discrete:
          num_outputs = action_space.n
          self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.is_continuous:
          num_outputs = action_space.shape[0]
          self.dist = DiagGaussian(self.base.output_size, num_outputs) # Diagonal Multivariate Gaussian
        else:
          raise NotImplementedError
      else:
        pass


    def act(self, inputs, states, masks, deterministic=False):
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, states
  '''

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

  def _train(self, *args, **kwargs):
    return self.train_episodically(*args, **kwargs)

  # endregion

  # region Abstract

  @abstractmethod
  def _sample_model(self, state, *args, **kwargs) -> Any:
    raise NotImplementedError

  @abstractmethod
  def train_episodically(self, *args, **kwargs):
    raise NotImplementedError

  # endregion
