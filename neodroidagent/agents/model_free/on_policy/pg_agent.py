#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from logging import warning

from draugr.torch_utilities.to_tensor import to_tensor
from draugr.writers import MockWriter
from draugr.writers.writer import Writer
from neodroid.environments.environment import Environment
from neodroid.interfaces.unity_specifications import EnvironmentSnapshot
from neodroidagent.agents.model_free.on_policy.policy_agent import PolicyAgent

from neodroidagent.exceptions.exceptions import NoTrajectoryException

from neodroidagent.training.procedures import train_episodically


__author__ = 'Christian Heider Nielsen'

from itertools import count

import numpy
import torch
from tqdm import tqdm

tqdm.monitor_interval = 0


class PGAgent(PolicyAgent):
  '''
  REINFORCE, Vanilla Policy Gradient method


  '''


  # region Protected

  def _optimise(self, loss, **kwargs):
    self.optimiser.zero_grad()
    loss.backward()
    if self._grad_clip:
      for params in self._distribution_regressor.parameters():
        params.grad.data.clamp_(self._grad_clip_low, self._grad_clip_high)
    self.optimiser.step()

  def _sample_model(self, state, **kwargs) -> tuple:
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

  def _sample(self, state, *args, **kwargs) -> tuple:
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

  def rollout(self,
              initial_state: EnvironmentSnapshot,
              environment: Environment,
              render: bool = False,
              metric_writer: Writer = MockWriter(),
              train: bool = True,
              max_length: int = None,
              disable_stdout: bool = False,
              **kwargs):
    '''Perform a single rollout until termination in environment

    :param disable_stdout:
    :param metric_writer:
    :type max_length: int
    :param max_length:
    :type train: bool
    :type render: bool
    :param initial_state: The initial state observation in the environment
    :param environment: The environment the agent interacts with
    :param render: Whether to render environment interaction
    :param train: Whether the agent should use the rollout to update its model
    :param kwargs:
    :return:
      -episode_signal (:py:class:`float`) - first output
      -episode_length-
      -average_episode_entropy-
    '''

    episode_signal = []
    episode_length = 0
    episode_entropy = []

    state = initial_state.observables

    '''
    with draugr.scroll_plot_class(self._distribution_regressor.output_shape,
                                  render=render,
                                  window_length=66) as s:
                                  '''
    for t in tqdm(count(1), f'Update #{self._update_i}', leave=False, disable=disable_stdout):
      action, action_log_prob, entropy = self.sample(state)

      snapshot = environment.react(action)

      state, signal, terminated = snapshot.observables, snapshot.signal, snapshot.terminated

      if self._signal_clipping:
        signal = numpy.clip(signal,
                            self._signal_clip_low,
                            self._signal_clip_high)

      episode_signal.append(signal)
      episode_entropy.append(entropy.to('cpu').numpy())
      if train:
        self._trajectory_trace.add_point(signal, action_log_prob, entropy)

      if render:
        environment.render()
        # s.draw(to_one_hot(self._distribution_regressor.output_shape, action)[0])

      if numpy.array(terminated).all() or (max_length and t > max_length):
        episode_length = t
        break

    if train:
      self.update()

    ep = numpy.array(episode_signal).sum(axis=0).mean()
    el = episode_length
    ee = numpy.array(episode_entropy).mean(axis=0).mean()

    if metric_writer:
      metric_writer.scalar('duration', el, self._update_i)
      metric_writer.scalar('signal', ep, self._update_i)
      metric_writer.scalar('entropy', ee, self._update_i)

    return ep, el, ee

  def infer(self, env, render=True):

    for episode_i in count(1):
      print('Episode {}'.format(episode_i))
      state = env.reset()

      for frame_i in count(1):

        action, *_ = self.sample(state)
        state, signal, terminated, info = env.act(action)
        if render:
          env.render()

        if terminated:
          break

  # endregion


# region Test
def pg_test(rollouts=None, skip=True):
  from neodroidagent.training.agent_session_entry_point import agent_session_entry_point
  from neodroidagent.training.sessions.parallel_training import parallelised_training
  import neodroidagent.configs.agent_test_configs.pg_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  agent_session_entry_point(PGAgent,
                            C,
                            parse_args=False,
                            training_session=parallelised_training,
                            skip_confirmation=skip)


def pg_run(rollouts=None, skip=True):
  from neodroidagent.training.agent_session_entry_point import agent_session_entry_point
  from neodroidagent.training.sessions.parallel_training import parallelised_training
  import neodroidagent.configs.agent_test_configs.pg_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  C.CONNECT_TO_RUNNING = True

  agent_session_entry_point(PGAgent,
                            C,
                            training_session=parallelised_training(training_procedure=train_episodically),
                            skip_confirmation=skip)


if __name__ == '__main__':
  # pg_test()
  pg_run()
# endregion
