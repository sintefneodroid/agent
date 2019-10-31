#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from logging import warning
from typing import Sequence

from draugr.torch_utilities.to_tensor import to_tensor
from draugr.writers import MockWriter
from neodroidagent.agents.torch_agents.model_free.on_policy.policy_agent import PolicyAgent
from neodroidagent.exceptions.exceptions import NoTrajectoryException

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 23/09/2019
           '''

import numpy
import torch
from tqdm import tqdm

tqdm.monitor_interval = 0


class PGAgent(PolicyAgent):
  '''
  REINFORCE, Vanilla Policy Gradient method


  '''

  def _remember(self,
                *,
                signal,
                action_log_prob,
                entropy,
                **kwargs):
    self._trajectory_trace.add_point(signal, action_log_prob, entropy)

  # region Protected

  def _optimise(self, loss, **kwargs):
    self.optimiser.zero_grad()
    loss.backward()
    if self._grad_clip:
      for params in self._distribution_regressor.parameters():
        params.grad.data.clamp_(self._grad_clip_low, self._grad_clip_high)
    self.optimiser.step()

  def _sample_model(self, state: Sequence, **kwargs) -> tuple:
    model_input = to_tensor(state, device=self._device, dtype=self._state_type)

    distributions = self._distribution_regressor(model_input)
    action, log_prob = self._distribution_regressor.sample(distributions)

    with torch.no_grad():
      entropy = self._distribution_regressor.entropy(distributions)

    return action, log_prob, entropy

  def _update(self,
              *,
              metric_writer=MockWriter(),
              **kwargs):
    '''

    :param metric_writer:
    :param args:
    :param kwargs:

    :returns:
    '''

    error = self.evaluate()

    if metric_writer:
      metric_writer.scalar('Error', error.detach().to('cpu').numpy())

    if error is not None:
      if self._use_batched_updates:
        self._accumulated_error += error
        if self._update_i % self._batch_size == 0:
          self._optimise(self._accumulated_error / self._batch_size)
          self._accumulated_error = to_tensor(0.0, device=self._device)
      else:
        self._optimise(error)

  def _sample(self, state: Sequence, *args, **kwargs) -> tuple:
    return self._sample_model(state)

  # endregion

  # region Public

  def evaluate(self, **kwargs):
    if not len(self._trajectory_trace) > 0:
      raise NoTrajectoryException

    trajectory = self._trajectory_trace.retrieve_trajectory()
    t_signals = trajectory.signal
    log_probs = trajectory.log_prob
    entropies = trajectory.entropy
    self._trajectory_trace.clear()

    ret = numpy.zeros_like(t_signals[0])
    policy_loss = []
    signals = []

    for r in t_signals[::-1]:
      ret = r + self._discount_factor * ret
      signals.insert(0, ret)

    signals = to_tensor(signals, device=self._device, dtype=self._signals_tensor_type)

    if signals.shape[0] > 1:
      stddev = signals.std()
      signals = (signals - signals.mean()) / (stddev + self._divide_by_zero_safety)
    elif signals.shape[0] == 0:
      warning(f'No signals received, got signals.shape[0]: {signals.shape[0]}')

    for log_prob, signal, entropy in zip(log_probs, signals, entropies):
      maximisation_term = signal + self._policy_entropy_regularisation * entropy
      expected_reward = -log_prob * maximisation_term
      policy_loss.append(expected_reward)

    if len(policy_loss[0].shape) < 1:
      loss = torch.stack(policy_loss).sum()
    else:
      loss = torch.cat(policy_loss).sum()
    return loss

  # endregion


# region Test


def pg_run(rollouts=None, skip=True, connect_to_running=True):
  from neodroidagent.sessions.session_entry_point import session_entry_point
  from neodroidagent.sessions.single_agent.parallel import ParallelSession
  import neodroidagent.configs.agent_test_configs.pg_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  C.CONNECT_TO_RUNNING = connect_to_running

  session_entry_point(PGAgent,
                      C,
                      session=ParallelSession,
                      skip_confirmation=skip)


def pg_test():
  pg_run(connect_to_running=False)


if __name__ == '__main__':
  pg_test()

# endregion
