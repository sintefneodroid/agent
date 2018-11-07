#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import draugr

from neodroid import EnvironmentState

__author__ = 'cnheider'

from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

import utilities as U
from agents.abstract.policy_agent import PolicyAgent

tqdm.monitor_interval = 0


class PGAgent(PolicyAgent):
  '''
  REINFORCE, Vanilla Policy Gradient method
  '''

  # region Private

  def __defaults__(self) -> None:

    self._policy_arch = U.CategoricalMLP
    self._accumulated_error = U.to_tensor(0.0, device=self._device)
    self._evaluation_function = torch.nn.CrossEntropyLoss()
    self._trajectory_trace = U.TrajectoryTraceBuffer()

    self._policy_arch_params = U.ConciseArchSpecification(**{
      'input_size':   None,  # Obtain from environment
      'hidden_layers':[64, 32, 16],
      'output_size':  None,  # Obtain from environment
      'activation':   F.relu,
      'use_bias':     True,
      })

    self._use_cuda = False
    self._discount_factor = 0.99
    self._use_batched_updates = False
    self._batch_size = 5
    self._pg_entropy_reg = 1e-4
    self._signal_clipping = False

    self._optimiser_learning_rate = 1e-4
    self._optimiser_type = torch.optim.Adam
    self._optimiser_weight_decay = 1e-5

    self._state_type = torch.float
    self._signals_tensor_type = torch.float

  # endregion

  # region Protected

  def _sample_model(self, state, **kwargs):
    state_tensor = U.to_tensor([state], device=self._device, dtype=self._state_type)
    with torch.no_grad():
      probs = self._policy(state_tensor)
    m = Categorical(probs)
    action = m.sample()
    return action.item()

  def _build(self, **kwargs) -> None:

    policy = self._policy_arch(
        **(self._policy_arch_params._asdict())
        ).to(self._device)

    self.optimiser = self._optimiser_type(
        policy.parameters(),
        lr=self._optimiser_learning_rate,
        weight_decay=self._optimiser_weight_decay,
        )

    self._policy = policy

  def _optimise_wrt(self, loss, **kwargs):
    self.optimiser.zero_grad()
    loss.backward()
    for params in self._policy.parameters():
      params.grad.data.clamp_(-1, 1)
    self.optimiser.step()

  # endregion

  # region Public

  def sample_action(self, state, discrete=True, **kwargs):
    if discrete:
      return self.sample_discrete_action(state)

    return self.sample_continuous_action(state)

  def sample_discrete_action(self, state):
    state_var = U.to_tensor([state], device=self._device, dtype=self._state_type)

    probs = self._policy(state_var)

    # action = np.argmax(probs)

    m = Categorical(probs)
    action_sample = m.sample()
    action = action_sample.item()

    return action, m.log_prob(action_sample), m.entropy()

  def sample_continuous_action(self, state):
    model_input = U.to_tensor([state], device=self._device, dtype=self._state_type)

    with torch.no_grad():
      mu, sigma_sq = self._policy(model_input)

      mu, sigma_sq = mu[0], sigma_sq[0]

    # std = self.sigma.exp().expand_as(mu)
    # dist = torch.Normal(mu, std)
    # return dist, value

    eps = torch.randn(mu.size())
    # calculate the probability
    action = (mu + sigma_sq.sqrt() * eps).data
    prob = U.normal(action, mu, sigma_sq)
    entropy = -0.5 * ((
                          sigma_sq
                          + 2
                          * U.pi_torch(self._device).expand_as(sigma_sq)
                      ).log()
                      + 1
                      )

    log_prob = prob.log()
    return action, log_prob, entropy

  def evaluate(self, **kwargs):
    R = 0
    policy_loss = []
    signals = []

    trajectory = self._trajectory_trace.retrieve_trajectory()
    t_signal = trajectory.signal
    log_probs = trajectory.log_prob
    entrp = trajectory.entropy
    self._trajectory_trace.clear()

    for r in t_signal[::-1]:
      R = r + self._discount_factor * R
      signals.insert(0, R)

    signals = U.to_tensor(signals, device=self._device, dtype=self._signals_tensor_type)

    if signals.shape[0] > 1:
      stddev = signals.std()
      signals = (signals - signals.mean()) / (stddev + self._divide_by_zero_safety)

    for log_prob, signal, entropy in zip(log_probs, signals, entrp):
      policy_loss.append(-log_prob * signal - self._pg_entropy_reg * entropy)

    loss = torch.cat(policy_loss).sum()
    return loss

  def update(self, *args, **kwargs):
    error = self.evaluate()

    if error is not None:
      if self._use_batched_updates:
        self._accumulated_error += error
        if self._rollout_i % self._batch_size == 0:
          self._optimise_wrt(self._accumulated_error / self._batch_size)
          self._accumulated_error = U.to_tensor(0.0, device=self._device)
      else:
        self._optimise_wrt(error)

  def rollout(self, initial_state, environment, render=False, train=True, **kwargs):
    if train:
      self._rollout_i += 1

    episode_signal = 0
    episode_length = 0
    episode_entropy = 0

    if type(initial_state) is EnvironmentState:
      state = initial_state.observables
    else:
      state = initial_state

    T = count(1)
    T = tqdm(T, f'Rollout #{self._rollout_i}', leave=False)

    for t in T:
      action, action_log_probs, entropy, *_ = self.sample_action(state)

      state, signal, terminated, info = environment.step(action)

      if self._signal_clipping:
        signal = np.clip(signal, -1.0, 1.0)

      episode_signal += signal
      episode_entropy += entropy.data.cpu().numpy()
      if train:
        self._trajectory_trace.add_trace(signal, action_log_probs, entropy)

      if render:
        environment.render()

      if terminated:
        episode_length = t
        break

    if train:
      self.update()

    avg_entropy = episode_entropy.mean().item()

    return episode_signal, episode_length, avg_entropy

  def infer(self, env, render=True):

    for episode_i in count(1):
      print('Episode {}'.format(episode_i))
      state = env.reset()

      for frame_i in count(1):

        action, *_ = self.sample_action(state)
        state, reward, terminated, info = env.step(action)
        if render:
          env.render()

        if terminated:
          break

  def train_episodically(
      self,
      env,
      rollouts=2000,
      render=False,
      render_frequency=100,
      stat_frequency=10,
      ):

    E = range(1, rollouts)
    E = tqdm(E, f'Episode: {1}', leave=False)

    stats = U.StatisticCollection(stats=('signal', 'duration', 'entropy'))

    for episode_i in E:
      initial_state = env.reset()

      if episode_i % stat_frequency == 0:
        draugr.styled_terminal_plot_stats_shared_x(stats,
                                          printer=E.write)

        E.set_description(f'Episode: {episode_i}, Running signal: {stats.signal.running_value[-1]}, '
                          f'Running length: {stats.duration.running_value[-1]}')

      if render and episode_i % render_frequency == 0:
        signal, dur, entrp, *extras = self.rollout(
            initial_state, env, render=render
            )
      else:
        signal, dur, entrp, *extras = self.rollout(initial_state, env)

      stats.duration.append(dur)
      stats.signal.append(signal)
      stats.entropy.append(entrp)

      if self._end_training:
        break

    return self._policy, stats
  # endregion


if __name__ == '__main__':
  import configs.agent_test_configs.test_pg_config as C

  U.test_agent_main(PGAgent, C)
