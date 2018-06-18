#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod

__author__ = 'cnheider'

import torch

import utilities as U
from agents.abstract.agent import Agent


class PolicyAgent(Agent):
  '''
All policy iteration agents should inherit from this class
'''

  def __init__(self, *args, **kwargs):
    self._policy_arch = None
    self._policy_arch_params = None
    self._policy = None
    super().__init__(*args, **kwargs)

  def _infer_input_output_sizes(self, env, **kwargs):
    super()._infer_input_output_sizes(env)
    if self._policy_arch_params is U.ConciseArchSpecification:
      di = self._policy_arch_params._asdict()
    else:
      di = self._policy_arch_params
    di['input_size'] = self._input_size
    di['output_size'] = self._output_size

    self._policy_arch_params = U.ConciseArchSpecification(**di)

  @abstractmethod
  def train_episodic(self, *args, **kwargs):
    raise NotImplementedError

  def _train(self, *args, **kwargs):
    return self.train_episodic(*args, **kwargs)

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
