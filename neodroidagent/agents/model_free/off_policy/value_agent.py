#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import math
import random
from abc import abstractmethod
from typing import Any, Dict, Sequence

import numpy
import torch

from draugr.torch_utilities import copy_state
from draugr.visualisation import sprint
from draugr.writers import MockWriter, TensorBoardPytorchWriter
from draugr.writers.writer import Writer
from neodroid.environments.environment import Environment
from neodroidagent.architectures.mock import MockArchitecture
from neodroidagent.interfaces.architecture import Architecture
from neodroidagent.interfaces.specifications import ExplorationSpecification
from neodroidagent.interfaces.torch_agent import TorchAgent
from warg.gdkc import GDKC

__author__ = 'cnheider'


class ValueAgent(TorchAgent):
  '''
All value iteration agents should inherit from this class
'''

  # region Public

  def __init__(self, *args, **kwargs):
    self._exploration_spec = ExplorationSpecification(start=0.99, end=0.04, decay=10000)
    self._initial_observation_period = 0

    self._value_arch_spec: GDKC = None
    self._value_model: Architecture = MockArchitecture()
    self._target_value_model: Architecture = MockArchitecture()

    self._optimiser_spec: GDKC = None

    self._naive_max_policy = False

    super().__init__(*args, **kwargs)

  def __build__(self, env: Environment, writer: Writer = None, print_model_repr=True, **kwargs):

    self._value_model = self._value_arch_spec().to(self._device)

    self._target_value_model = self._value_arch_spec().to(self._device)
    self._target_value_model: Architecture = copy_state(
        target=self._target_value_model,
        source=self._value_model)
    self._target_value_model.eval()

    self._optimiser = self._optimiser_spec(self._value_model.parameters())

    if writer:
      dummy_in = torch.rand(1, *self.input_shape)

      model = copy.deepcopy(self._value_model)
      model.to('cpu')

      if isinstance(writer, TensorBoardPytorchWriter):
        writer.graph(model, dummy_in)

    if print_model_repr:
      sprint(f'Value model: {self._value_model}',
             highlight=True,
             color='cyan')

  @property
  def models(self) -> Dict[str, Architecture]:
    return {'_value_model':self._value_model}

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

  # endregion

  # region Abstract

  def _sample(self,
              state: Sequence,
              no_random=False,
              metric_writer: Writer = MockWriter(),
              **kwargs):
    self._sample_i += 1
    s = self.epsilon_random_exploration(self._sample_i)
    if metric_writer:
      metric_writer.scalar('Current Eps Threshold', self._current_eps_threshold, self._sample_i)

    if ((s and self._sample_i > self._initial_observation_period) or
        no_random):

      return self._sample_model(state)

    return self._sample_random_process(state)

  @abstractmethod
  def _sample_model(self, state, **kwargs) -> Any:
    raise NotImplementedError

  # endregion

  # region Protected

  def _post_io_inference(self, env):
    self._value_arch_spec.kwargs['input_shape'] = self._input_shape
    self._value_arch_spec.kwargs['output_shape'] = self._output_shape

  def _sample_random_process(self, state):
    r = numpy.arange(self._output_shape[0])
    sample = numpy.random.choice(r, len(state))
    return sample

  # endregion
