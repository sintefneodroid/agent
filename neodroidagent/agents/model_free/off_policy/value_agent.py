#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import math
import random
from abc import abstractmethod
from typing import Any, Dict, Sequence

import numpy
import torch
from torch.nn.functional import smooth_l1_loss

from draugr.torch_utilities import copy_state
from draugr.visualisation import sprint
from draugr.writers import MockWriter, TensorBoardPytorchWriter
from draugr.writers.writer import Writer

from neodroid.interfaces.spaces import ObservationSpace, ActionSpace, SignalSpace
from neodroidagent.architectures import MLP
from neodroidagent.architectures.mock import MockArchitecture
from neodroidagent.interfaces.architecture import Architecture
from neodroidagent.interfaces.specifications import ExplorationSpecification
from neodroidagent.interfaces.torch_agent import TorchAgent
from neodroidagent.memory import ReplayBuffer
from warg.gdkc import GDKC
from warg.kw_passing import passes_kws_to_super_init

__author__ = 'Christian Heider Nielsen'

@passes_kws_to_super_init(super_base=TorchAgent)
class ValueAgent(TorchAgent):
  '''
All value iteration agents should inherit from this class
'''

  # region Private

  def __init__(self,
               exploration_spec=ExplorationSpecification(start=0.99, end=0.04, decay=10000),
               initial_observation_period=0,
               value_model: Architecture = MockArchitecture(),
               target_value_model: Architecture = MockArchitecture(),
               naive_max_policy=False,
               memory_buffer=ReplayBuffer(10000),
               # self._memory = U.PrioritisedReplayMemory(config.REPLAY_MEMORY_SIZE)  # Cuda trouble
               evaluation_function=smooth_l1_loss,
               value_arch_spec: Architecture = GDKC(MLP,
                                                    input_shape=None,  # Obtain from environment
                                                    hidden_layers=None,
                                                    output_shape=None  # Obtain from environment
                                                    ),
               batch_size=128,
               discount_factor=0.95,
               learning_frequency=1,
               sync_target_model_frequency=1000,
               state_type=torch.float,
               value_type=torch.float,
               action_type=torch.long,
               use_double_dqn=True,
               clamp_gradient=False,
               signal_clipping=True,
               early_stopping_condition=None,
               optimiser_spec=GDKC(torch.optim.RMSprop,
                                   alpha=0.9,
                                   lr=0.0025,
                                   eps=1e-02,
                                   momentum=0.0),
               **kwargs):
    '''

    :param exploration_spec:
    :param initial_observation_period:
    :param value_model:
    :param target_value_model:
    :param naive_max_policy:
    :param memory_buffer:
    :param evaluation_function:
    :param value_arch_spec:
    :param batch_size:
    :param discount_factor:
    :param learning_frequency:
    :param sync_target_model_frequency:
    :param state_type:
    :param value_type:
    :param action_type:
    :param use_double_dqn:
    :param clamp_gradient:
    :param signal_clipping:
    :param early_stopping_condition:
    :param optimiser_spec:
    :param kwargs:
    '''
    super().__init__(**kwargs)
    self._exploration_spec = exploration_spec
    self._initial_observation_period = initial_observation_period
    self._value_model: Architecture = value_model
    self._target_value_model: Architecture = target_value_model
    self._naive_max_policy = naive_max_policy
    self._memory_buffer = memory_buffer
    # self._memory = U.PrioritisedReplayMemory(config.REPLAY_MEMORY_SIZE)  # Cuda trouble
    self._evaluation_function = evaluation_function
    self._value_arch_spec: Architecture = value_arch_spec
    self._batch_size = batch_size
    self._discount_factor = discount_factor
    self._learning_frequency = learning_frequency
    self._initial_observation_period = initial_observation_period
    self._sync_target_model_frequency = sync_target_model_frequency
    self._state_type = state_type
    self._value_type = value_type
    self._action_type = action_type
    self._use_double_dqn = use_double_dqn
    self._clamp_gradient = clamp_gradient
    self._signal_clipping = signal_clipping
    self._early_stopping_condition = early_stopping_condition
    self._optimiser_spec = optimiser_spec



  def __build__(self,
                observation_space: ObservationSpace,
                action_space: ActionSpace,
                signal_space: SignalSpace,
                writer: Writer = None,
                print_model_repr=True,
                **kwargs):

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

  # endregion

  # region Public
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

  def _post_io_inference(self, observation_space: ObservationSpace,
                action_space: ActionSpace,
                signal_space: SignalSpace):
    self._value_arch_spec.kwargs['input_shape'] = self._input_shape
    self._value_arch_spec.kwargs['output_shape'] = self._output_shape

  def _sample_random_process(self, state):
    r = numpy.arange(self._output_shape[0])
    sample = numpy.random.choice(r, len(state))
    return sample

  # endregion
