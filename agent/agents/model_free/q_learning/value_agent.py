#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
from abc import abstractmethod
from typing import Any, Iterable, Sequence

import draugr
from agent.interfaces.torch_agent import TorchAgent
from agent.interfaces.specifications import ExplorationSpecification, GDCS
from draugr.writers.writer import Writer



import math
import random

import numpy as np
import torch
from agent import utilities as U
from neodroid.environments.environment import Environment

__author__ = 'cnheider'

class ValueAgent(TorchAgent):
  '''
All value iteration agents should inherit from this class
'''

  # region Public

  def __init__(self, *args, **kwargs):
    self._exploration_spec = ExplorationSpecification(start=0.99, end=0.04, decay=10000)
    self._initial_observation_period = 0

    self._value_arch_spec: GDCS = None
    self._value_model = None

    self._naive_max_policy = False

    super().__init__(*args, **kwargs)

  def sample(self,
             state:Sequence,
             disallow_random_sample=False,
             stat_writer: Writer = None,
             **kwargs):
    self._step_i += 1
    s = self.epsilon_random_exploration(self._step_i)
    if stat_writer:
      stat_writer.scalar('Current Eps Threshold', self._current_eps_threshold, self._step_i)

    if ((s and self._step_i > self._initial_observation_period) or
        disallow_random_sample):

      return self._sample_model(state)

    return self._sample_random_process(state)

  def _build(self, env:Environment, stat_writer:Writer = None, **kwargs):
    if stat_writer:
      dummy_in = torch.rand(1, *self.input_shape)

      model = copy.deepcopy(self._value_model)
      model.to('cpu')

      if isinstance(stat_writer, draugr.TensorBoardXWriter):
        stat_writer._graph(model, dummy_in)

    num_params = sum(param.numel() for param in self._value_model.parameters())
    num_trainable_params = sum(
        p.numel() for p in self._value_model.parameters() if p.requires_grad)

    draugr.sprint(f'trainable/num_params: {num_trainable_params}/{num_params}\n', highlight=True,
                  color='cyan')

  @property
  def models(self):
    return (self._value_model,)

  def epsilon_random_exploration(self, steps_taken):
    '''
:param steps_taken:
:return:
'''
    assert 0 <= self._exploration_spec.end <= self._exploration_spec.start

    if steps_taken == 0:
      return False

    sample = random.random()

    a = self._exploration_spec.start - self._exploration_spec.end

    b = math.exp(-1. * steps_taken / (self._exploration_spec.decay + self._divide_by_zero_safety))
    self._current_eps_threshold = self._exploration_spec.end + a * b

    return sample > self._current_eps_threshold

  def save(self, C):
    U.save_model(self._value_model, **C)

  def load(self, model_path, evaluation):
    print('Loading latest model: ' + model_path)
    self._value_model = self._value_arch_spec.constructor(**self._value_arch_spec.kwargs)
    self._value_model.load_state_dict(torch.load(model_path))
    if evaluation:
      self._value_model = self._value_model.eval()
      self._value_model.train(False)
    if self._use_cuda:
      self._value_model = self._value_model.cuda()
    else:
      self._value_model = self._value_model.cpu()

  # endregion

  # region Abstract

  @abstractmethod
  def _sample_model(self, state, **kwargs) -> Any:
    raise NotImplementedError

  # endregion

  # region Protected

  def _post_io_inference(self, env):
    self._value_arch_spec.kwargs['input_shape'] = self._input_shape
    self._value_arch_spec.kwargs['output_shape'] = self._output_shape

  def _sample_random_process(self, state):
    r = np.arange(self._output_shape[0])
    sample = np.random.choice(r, len(state))
    return sample

  # endregion
