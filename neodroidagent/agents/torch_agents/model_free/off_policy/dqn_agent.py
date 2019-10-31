#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

from draugr import copy_state
from draugr.torch_utilities.to_tensor import to_tensor
from draugr.writers import MockWriter
from neodroidagent.agents.torch_agents.model_free.off_policy.value_agent import ValueAgent
from neodroidagent.procedures.training.off_policy_episodic import OffPolicyEpisodic

__author__ = 'Christian Heider Nielsen'

import torch


class DQNAgent(ValueAgent):
  '''

'''

  def _remember(self,
                *,
                state,
                action,
                signal,
                next_state,
                terminated,
                **kwargs):
    self._memory_buffer.add_transition(state,
                                       action,
                                       signal,
                                       next_state,
                                       terminated
                                       )

  # region Protected

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
    model_input = to_tensor(state, device=self._device, dtype=self._state_type)

    with torch.no_grad():
      action_value_estimates = self._value_model(model_input)
      max_value_action_idx = action_value_estimates.max(-1)[1].to('cpu').numpy().tolist()

    return max_value_action_idx

  def _update(self, *, metric_writer=MockWriter(), **kwargs):
    if self._batch_size < len(self._memory_buffer):
      # indices, transitions = self._memory.sample_transitions(self.C.BATCH_SIZE)
      transitions = self._memory_buffer.sample_transitions(self._batch_size)

      td_error = self.evaluate(transitions)
      self._optimise(td_error)

      if metric_writer:
        metric_writer.scalar('td_error', td_error.mean().item())

      # self._memory.batch_update(indices, td_error.tolist())  # Cuda trouble
    else:
      logging.info('Batch size is larger than current memory size, skipping update')

    if self._use_double_dqn and self.update_i % self._sync_target_model_frequency == 0:
      self._target_value_model = copy_state(target=self._target_value_model,
                                            source=self._value_model)
      if metric_writer:
        metric_writer.scalar('Target Model Synced', self.update_i, self.update_i)

  # endregion

  # region Public

  def evaluate(self, batch, **kwargs):
    '''

:param batch:
:type batch:
:return:
:rtype:
'''
    states = to_tensor(batch.state,
                       dtype=self._state_type,
                       device=self._device).transpose(0, 1)

    true_signals = to_tensor(batch.signal,
                             dtype=self._value_type,
                             device=self._device).transpose(0, 1)

    action_indices = to_tensor(batch.action,
                               dtype=self._action_type,
                               device=self._device).transpose(0, 1)

    non_terminal_mask = to_tensor(batch.non_terminal_numerical,
                                  dtype=torch.float,
                                  device=self._device).transpose(0, 1)

    successor_states = to_tensor(batch.successor_state,
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
    p = self._value_model(states)
    Q_state = p.gather(-1, action_indices).squeeze(-1)

    return self._loss_function(Q_state, Q_expected)

  def infer(self, state, **kwargs):
    model_input = to_tensor(state, device=self._device, dtype=self._state_type)
    with torch.no_grad():
      value = self._value_model(model_input)
    return value

  def step(self, state, env):
    action = self.sample(state)
    return action, env.react(action)

  # endregion


# region Test

def dqn_run(rollouts=None, skip=True, connect_to_running=True):
  from neodroidagent.sessions.session_entry_point import session_entry_point
  from neodroidagent.sessions.single_agent.parallel import ParallelSession
  import neodroidagent.configs.agent_test_configs.dqn_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  C.CONNECT_TO_RUNNING = connect_to_running

  session_entry_point(DQNAgent,
                      C,
                      session=ParallelSession(procedure=OffPolicyEpisodic,
                                              auto_reset_on_terminal_state=True),
                      skip_confirmation=skip)


def dqn_test():
  dqn_run(rollouts=None, skip=True, connect_to_running=False)


if __name__ == '__main__':
  dqn_test()
# endregion
