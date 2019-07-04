#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

from agent.architectures import MLP
from agent.interfaces.partials.agents.torch_agents.value_agent import ValueAgent
from agent.interfaces.specifications.generalised_delayed_construction_specification import GDCS
from agent.memory import ReplayBuffer
from agent.training.train_agent import parallelised_training, train_agent
from warg.named_ordered_dictionary import NOD, NamedOrderedDictionary

__author__ = 'cnheider'
from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from agent import utilities as U


class DQNAgent(ValueAgent):
  '''

'''

  # region Protected

  def __defaults__(self) -> None:
    self._memory_buffer = ReplayBuffer(10000)
    # self._memory = U.PrioritisedReplayMemory(config.REPLAY_MEMORY_SIZE)  # Cuda trouble

    self._use_cuda = False

    self._evaluation_function = F.smooth_l1_loss

    self._value_arch_spec = GDCS(MLP, NamedOrderedDictionary({
      'input_shape':            None,  # Obtain from environment
      'hidden_layers':          None,
      'output_shape':           None,  # Obtain from environment
      'hidden_layer_activation':torch.tanh,
      'use_bias':               True,
      }))

    self._batch_size = 128

    self._discount_factor = 0.95
    self._learning_frequency = 1
    self._initial_observation_period = 0
    self._sync_target_model_frequency = 1000

    self._state_type = torch.float
    self._value_type = torch.float
    self._action_type = torch.long

    self._use_double_dqn = True
    self._clamp_gradient = False
    self._signal_clipping = True

    self._early_stopping_condition = None
    self._target_value_model = None

    self._optimiser_spec = GDCS(torch.optim.RMSprop, kwargs=NOD(alpha=0.9,
                                                                lr=0.0025,
                                                                eps=1e-02,
                                                                momentum=0.0))
    self._optimiser = None

  def _build(self, **kwargs) -> None:

    self._value_model = self._value_arch_spec.constructor(**self._value_arch_spec.kwargs).to(self._device)

    self._target_value_model = self._value_arch_spec.constructor(**self._value_arch_spec.kwargs).to(
        self._device)
    self._target_value_model = U.copy_state(target=self._target_value_model, source=self._value_model)
    self._target_value_model.eval()

    self._optimiser = self._optimiser_spec.constructor(self._value_model.parameters(),
                                                       **self._optimiser_spec.kwargs
                                                       )

  def _optimise(self, error, **kwargs):
    '''

:param error:
:type error:
:return:
'''
    self._optimiser.zero_grad()
    error.backward()
    if self._clamp_gradient:
      for params in self._value_model.parameters():
        params.grad.data.clamp_(-1, 1)
    self._optimiser.step()

  def _sample_model(self, state, **kwargs):
    model_input = U.to_tensor(state, device=self._device, dtype=self._state_type)

    with torch.no_grad():
      action_value_estimates = self._value_model(model_input)

    max_value_action_idx = action_value_estimates.max(-1)[1].to('cpu').numpy().tolist()
    return max_value_action_idx

  # endregion

  # region Public

  def evaluate(self, batch, **kwargs):
    '''

:param batch:
:type batch:
:return:
:rtype:
'''
    states = U.to_tensor(batch.state,
                         dtype=self._state_type,
                         device=self._device).transpose(0, 1)

    true_signals = U.to_tensor(batch.signal,
                               dtype=self._value_type,
                               device=self._device).transpose(0, 1)

    action_indices = U.to_tensor(batch.action,
                                 dtype=self._action_type,
                                 device=self._device).transpose(0, 1)

    non_terminal_mask = U.to_tensor(batch.non_terminal_numerical,
                                    dtype=torch.float,
                                    device=self._device).transpose(0, 1)

    successor_states = U.to_tensor(batch.successor_state,
                                   dtype=self._state_type,
                                   device=self._device).transpose(0, 1)

    # Calculate Q of successors
    with torch.no_grad():
      Q_successors = self._value_model(successor_states)

    Q_successors_max_action_indices = Q_successors.max(-1)[1]
    Q_successors_max_action_indices = Q_successors_max_action_indices.unsqueeze(-1)
    if self._use_double_dqn:
      with torch.no_grad():
        Q_successors = self._target_value_model(successor_states)

    max_next_values = Q_successors.gather(-1, Q_successors_max_action_indices).squeeze(-1)
    # a = Q_max_successor[non_terminal_mask]
    Q_max_successor = max_next_values * non_terminal_mask

    # Integrate with the true signal
    Q_expected = true_signals + (self._discount_factor * Q_max_successor)

    # Calculate Q of state
    action_indices = action_indices.unsqueeze(-1)
    Q_state = self._value_model(states).gather(-1, action_indices).squeeze(-1)

    return self._evaluation_function(Q_state, Q_expected)

  def update_models(self, *, stat_writer=None, **kwargs):
    if self._batch_size < len(self._memory_buffer):
      # indices, transitions = self._memory.sample_transitions(self.C.BATCH_SIZE)
      transitions = self._memory_buffer.sample_transitions(self._batch_size)

      td_error = self.evaluate(transitions)
      self._optimise(td_error)

      if stat_writer:
        stat_writer.scalar('td_error', td_error.mean().item())

      # self._memory.batch_update(indices, td_error.tolist())  # Cuda trouble
    else:
      logging.info('Batch size is larger than current memory size, skipping update')

  def rollout(self,
              initial_state,
              environment,
              *,
              render=False,
              stat_writer=None,
              train=True,
              disallow_random_sample=False,
              **kwargs):
    self._update_i += 1

    state = initial_state
    episode_signal = []
    episode_length = []

    T = count(1)
    T = tqdm(T, f'Rollout #{self._update_i}', leave=False, disable=not render)

    for t in T:
      self._step_i += 1

      action = self.sample_action(state, disallow_random_sample=disallow_random_sample)
      next_state, signal, terminated, *_ = environment.react(action).to_gym_like_output()

      if render:
        environment.render()

      if self._signal_clipping:
        signal = np.clip(signal, -1.0, 1.0)

      self._memory_buffer.add_transition(state,
                                         action,
                                         signal,
                                         next_state,
                                         terminated
                                         )

      td_error = 0

      if (len(self._memory_buffer) >= self._batch_size
          and self._step_i > self._initial_observation_period
          and self._step_i % self._learning_frequency == 0
      ):

        self.update_models()

      if self._use_double_dqn and self._step_i % self._sync_target_model_frequency == 0:
        self._target_value_model = U.copy_state(target=self._target_value_model, source=self._value_model)
        if stat_writer:
          stat_writer.scalar('Target Model Synced', self._step_i, self._step_i)

      episode_signal.append(signal)

      if terminated.all():
        episode_length = t
        break

      state = next_state

    ep = np.array(episode_signal).sum(axis=0).mean()
    el = episode_length

    if stat_writer:
      stat_writer.scalar('duration', el, self._update_i)
      stat_writer.scalar('signal', ep, self._update_i)
      stat_writer.scalar('current_eps_threshold', self._current_eps_threshold, self._update_i)

    return ep, el

  def infer(self, state, **kwargs):
    model_input = U.to_tensor(state, device=self._device, dtype=self._state_type)
    with torch.no_grad():
      value = self._value_model(model_input)
    return value

  def step(self, state, env):
    action = self.sample_action(state)
    return action, env.react(action)

  # endregion


# region Test


def dqn_test(rollouts=None, skip=True):
  import agent.configs.agent_test_configs.dqn_test_config as C

  # import configs.cnn_dqn_config as C
  if rollouts:
    C.ROLLOUTS = rollouts

  train_agent(DQNAgent,
              C,
              parse_args=False,
              training_session=parallelised_training,
              skip_confirmation=skip)
  # test_cnn_dqn_agent(C)


def dqn_run(rollouts=None, skip=True):
  import agent.configs.agent_test_configs.dqn_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  C.CONNECT_TO_RUNNING = True

  train_agent(DQNAgent,
              C,
              training_session=parallelised_training,
              skip_confirmation=skip)


if __name__ == '__main__':
  dqn_test()
# endregion
