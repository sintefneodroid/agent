#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from warnings import warn

import draugr
from draugr import TensorBoardWriter
from neodroid.models import EnvironmentState
from agent.utilities.exceptions.exceptions import NoTrajectoryException
from warg import NOD

from agent.architectures import CategoricalMLP
from agent.procedures.train_agent import agent_test_main, parallel_train_agent_procedure
from agent.utilities.specifications.generalised_delayed_construction_specification import GDCS
from agent.utilities.specifications.training_resume import TR

__author__ = 'cnheider'

from itertools import count

import numpy as np
import torch
from torch.distributions import Categorical, Normal
from tqdm import tqdm

from agent import utilities as U
from agent.agents.abstract.policy_agent import PolicyAgent

tqdm.monitor_interval = 0


class PGAgent(PolicyAgent):
  '''
  REINFORCE, Vanilla Policy Gradient method

  See method __defaults__ for default parameters
  '''

  # region Private

  def __defaults__(self) -> None:

    self._accumulated_error = U.to_tensor(0.0, device=self._device)
    self._evaluation_function = torch.nn.CrossEntropyLoss()
    self._trajectory_trace = U.TrajectoryTraceBuffer()

    self._policy_arch_spec = GDCS(CategoricalMLP, NOD(**{
      'input_size':             None,  # Obtain from environment
      'hidden_layers':          None,
      'output_size':            None,  # Obtain from environment
      'hidden_layer_activation':torch.relu,
      'use_bias':               True,
      }))

    self._use_cuda = False
    self._discount_factor = 0.99
    self._use_batched_updates = False
    self._batch_size = 5
    self._pg_entropy_reg = 1e-4
    self._signal_clipping = False
    self._signal_clip_low = -1.0
    self._signal_clip_high = -self._signal_clip_low

    self._optimiser_spec = GDCS(torch.optim.Adam, NOD(lr=1e-4, weight_decay=1e-5))

    self._state_type = torch.float
    self._signals_tensor_type = torch.float
    self._discrete = True
    self._grad_clip = False
    self._grad_clip_low = -1
    self._grad_clip_high = 1
    self._std = .3

  # endregion

  # region Protected

  def _build(self, **kwargs) -> None:
    self._policy_model = self._policy_arch_spec.constructor(**self._policy_arch_spec.kwargs).to(self._device)

    self.optimiser = self._optimiser_spec.constructor(self._policy_model.parameters(),
                                                      **self._optimiser_spec.kwargs)

  def _optimise_wrt(self, loss, **kwargs):
    self.optimiser.zero_grad()
    loss.backward()
    if self._grad_clip:
      for params in self._policy_model.parameters():
        params.grad.data.clamp_(self._grad_clip_low, self._grad_clip_high)
    self.optimiser.step()

  # endregion

  # region Public

  def sample_action(self, state, *args, **kwargs):
    return self._sample_model(state)

  def _sample_model(self, state, *args, **kwargs):

    if self._discrete:
      return self.sample_discrete_action(state)

    return self.sample_continuous_action(state)

  def sample_discrete_action(self, state):
    state_var = U.to_tensor(state, device=self._device, dtype=self._state_type)

    probs = self._policy_model(state_var)
    distribution = Categorical(logits=probs)
    action_sample = distribution.sample()
    log_prob = distribution.log_prob(action_sample)
    with torch.no_grad():
      action = action_sample.to('cpu').numpy()
      entropy = distribution.entropy()

    return action, log_prob, entropy

  def sample_continuous_action(self, state):
    model_input = U.to_tensor(state, device=self._device, dtype=self._state_type)

    mean, log_std = self._policy_model(model_input)

    std = log_std.exp().expand_as(mean)
    distribution = Normal(mean, std)
    action = distribution.sample()
    log_prob = distribution.log_prob(action)

    with torch.no_grad():
      entropy = distribution.entropy()  # .mean()
      action = action.to('cpu').numpy().tolist()

    '''eps = torch.randn(mean.size()).to(self._device)
    # calculate the probability
    a = mean + sigma_sq.sqrt() * eps
    action = a.data
    torch.distributions.Normal(mean,sigma_sq)
    
    
    prob = U.normal(action, mean, sigma_sq,device=self._device)
    entropy = -0.5 * ((sigma_sq
                       + 2
                       * U.pi_torch(self._device).expand_as(sigma_sq)
                       ).log()
                      + 1
                      )

    log_prob = prob.log()
    '''

    return action, log_prob, entropy

  def evaluate(self, **kwargs):
    if not len(self._trajectory_trace) > 0:
      raise NoTrajectoryException

    trajectory = self._trajectory_trace.retrieve_trajectory()
    t_signals = trajectory.signal
    log_probs = trajectory.log_prob
    entropies = trajectory.entropy
    self._trajectory_trace.clear()

    R = 0
    policy_loss = []
    signals = []

    for r in t_signals[::-1]:
      R = r + self._discount_factor * R
      signals.insert(0, R)

    signals = U.to_tensor(signals, device=self._device, dtype=self._signals_tensor_type)

    if signals.shape[0] > 1:
      stddev = signals.std()
      signals = (signals - signals.mean()) / (stddev + self._divide_by_zero_safety)
    elif signals.shape[0] == 0:
      warn(f'No signals received, got signals.shape[0]:{signals.shape[0]}')

    for log_prob, signal, entropy in zip(log_probs, signals, entropies):
      policy_loss.append(-log_prob * signal - self._pg_entropy_reg * entropy)

    loss = torch.cat(policy_loss).sum()
    return loss

  def update(self, *args, **kwargs):
    '''

    :param args:
    :param kwargs:

    :returns:
    '''

    error = self.evaluate()

    if error is not None:
      if self._use_batched_updates:
        self._accumulated_error += error
        if self._rollout_i % self._batch_size == 0:
          self._optimise_wrt(self._accumulated_error / self._batch_size)
          self._accumulated_error = U.to_tensor(0.0, device=self._device)
      else:
        self._optimise_wrt(error)

  def rollout(self,
              initial_state,
              environment,
              render: bool = False,
              train: bool = True,
              max_length: int = None,
              **kwargs):
    '''Perform a single rollout until termination in environment

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

    if train:
      self._rollout_i += 1

    episode_signal = []
    episode_length = 0
    episode_entropy = []

    if isinstance(initial_state, EnvironmentState):
      state = initial_state.observables
    else:
      state = initial_state

    T = count(1)
    T = tqdm(T, f'Rollout #{self._rollout_i}', leave=False)

    for t in T:
      action, action_log_probs, entropy, *_ = self.sample_action(state)

      state, signal, terminated, *_ = environment.react(action)

      if self._signal_clipping:
        signal = np.clip(signal, self._signal_clip_low, self._signal_clip_high)

      episode_signal.append(signal)
      episode_entropy.append(entropy.to('cpu').numpy())
      if train:
        self._trajectory_trace.add_trace(signal, action_log_probs, entropy)

      if render:
        environment.render()

      if np.array(terminated).all() or (max_length and t > max_length):
        episode_length = t
        break

    if train:
      self.update()

    ep = np.array(episode_signal).mean()
    el = episode_length
    ee = np.array(episode_entropy).mean()
    return ep, el, ee

  def infer(self, env, render=True):

    for episode_i in count(1):
      print('Episode {}'.format(episode_i))
      state = env.reset()

      for frame_i in count(1):

        action, *_ = self.sample_action(state)
        state, signal, terminated, info = env.act(action)
        if render:
          env.render()

        if terminated:
          break

  def train_episodically(self,
                         env,
                         test_env,
                         *,
                         rollouts=2000,
                         render=False,
                         render_frequency=100,
                         stat_frequency=10,
                         ) -> TR:

    E = range(1, rollouts)
    E = tqdm(E, f'Episode: {1}', leave=False)

    with TensorBoardWriter(str(self._log_directory)) as stat_writer:

      for episode_i in E:
        initial_state = env.reset()

        if render and episode_i % render_frequency == 0:
          signal, dur, entropy, *extras = self.rollout(initial_state, env, render=render)
        else:
          signal, dur, entropy, *extras = self.rollout(initial_state, env)

        stat_writer.scalar('duration', dur, episode_i)
        stat_writer.scalar('signal', signal, episode_i)
        stat_writer.scalar('entropy', entropy, episode_i)

        if self._end_training:
          break

    return TR(self._policy_model, None)

  def train_episodically_old(self,
                             env,
                             test_env,
                             *,
                             rollouts=2000,
                             render=False,
                             render_frequency=100,
                             stat_frequency=10,
                             ) -> TR:

    E = range(1, rollouts)
    E = tqdm(E, f'Episode: {1}', leave=False)

    stats = draugr.StatisticCollection(stats=('signal', 'duration', 'entropy'))

    for episode_i in E:
      initial_state = env.reset()

      if episode_i % stat_frequency == 0:
        draugr.styled_terminal_plot_stats_shared_x(stats,
                                                   printer=E.write)

        E.set_description(f'Epi: {episode_i}, '
                          f'Sig: {stats.signal.running_value[-1]:.3f}, '
                          f'Dur: {stats.duration.running_value[-1]:.1f}')

      if render and episode_i % render_frequency == 0:
        signal, dur, entropy, *extras = self.rollout(initial_state, env, render=render)
      else:
        signal, dur, entropy, *extras = self.rollout(initial_state, env)

      stats.duration.append(dur)
      stats.signal.append(signal)
      stats.entropy.append(entropy)

      if self._end_training:
        break

    return NOD(model=self._policy_model, stats=stats)
  # endregion


# region Test
def pg_test(rollouts=None):
  import agent.configs.agent_test_configs.pg_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  agent_test_main(PGAgent, C, parse_args=False, training_procedure=parallel_train_agent_procedure)


if __name__ == '__main__':
  pg_test()
# endregion
