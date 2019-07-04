#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from logging import warning

import draugr
from agent.architectures import CategoricalMLP
from agent.exceptions.exceptions import NoTrajectoryException
from agent.interfaces.partials.agents.torch_agents.policy_agent import PolicyAgent
from agent.interfaces.specifications.generalised_delayed_construction_specification import GDCS
from agent.memory import TrajectoryBuffer
from agent.training.procedures import train_episodically, to_tensor
from agent.training.train_agent import parallelised_training, train_agent
from neodroid.interfaces.environment_models import EnvironmentSnapshot
from neodroid.utilities.transformations.encodings import to_one_hot
from warg.named_ordered_dictionary import NOD

__author__ = 'cnheider'

from itertools import count

import numpy as np
import torch
from torch.distributions import Categorical, Normal
from tqdm import tqdm

from agent import utilities as U

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
    self._trajectory_trace = TrajectoryBuffer()

    self._policy_arch_spec = GDCS(CategoricalMLP, NOD(**{'input_shape':            None,
                                                         # Obtain from environment
                                                         'hidden_layers':          None,
                                                         'output_shape':           None,
                                                         # Obtain from environment
                                                         'hidden_layer_activation':torch.relu,
                                                         'use_bias':               True,
                                                         }))

    self._use_cuda = False
    self._discount_factor = 0.95
    self._use_batched_updates = False
    self._batch_size = 5
    self._policy_entropy_regularisation = 1
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
    self._distribution_parameter_regressor = self._policy_arch_spec.constructor(
        **self._policy_arch_spec.kwargs).to(self._device)

    self.optimiser = self._optimiser_spec.constructor(self._distribution_parameter_regressor.parameters(),
                                                      **self._optimiser_spec.kwargs)

  def _optimise(self, loss, **kwargs):
    self.optimiser.zero_grad()
    loss.backward()
    if self._grad_clip:
      for params in self._distribution_parameter_regressor.parameters():
        params.grad.data.clamp_(self._grad_clip_low, self._grad_clip_high)
    self.optimiser.step()

  def _sample_model(self, state, **kwargs):

    if self._policy_arch_spec.kwargs['discrete']:
      return self.sample_discrete_action(state)

    return self.sample_continuous_action(state)

  # endregion

  # region Public

  def sample_action(self, state, *args, **kwargs):
    return self._sample_model(state)

  def sample_discrete_action(self, state):
    state_var = U.to_tensor(state, device=self._device, dtype=self._state_type)

    probs = self._distribution_parameter_regressor(state_var)
    distribution = Categorical(logits=probs)
    action_sample = distribution.sample()
    log_prob = distribution.log_prob(action_sample)
    with torch.no_grad():
      action = action_sample.to('cpu').numpy()
      entropy = distribution.entropy()

    return action, log_prob, entropy, probs

  def sample_continuous_action(self, state):
    model_input = to_tensor(state, device=self._device, dtype=self._state_type)

    mean, log_std = self._distribution_parameter_regressor(model_input)

    std = log_std.exp().expand_as(mean)
    distribution = Normal(mean, std)
    action = distribution.sample()
    log_prob = distribution.log_prob(action)

    with torch.no_grad():
      entropy = distribution.entropy()  # .mean()
      action = action.to('cpu').numpy().tolist()

    '''
    eps = torch.randn(mean.size()).to(self._device)
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

    return action, log_prob, entropy, distribution

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
      warning(f'No signals received, got signals.shape[0]:{signals.shape[0]}')

    for log_prob, signal, entropy in zip(log_probs, signals, entropies):
      maximisation_term = signal + self._policy_entropy_regularisation * entropy
      expected_reward = - log_prob * maximisation_term
      policy_loss.append(expected_reward)

    loss = torch.cat(policy_loss).sum()
    return loss

  def update_models(self, *, stat_writer=None, **kwargs):
    '''

    :param stat_writer:
    :param args:
    :param kwargs:

    :returns:
    '''

    error = self.evaluate()

    if stat_writer:
      stat_writer.scalar('Error', error)

    if error is not None:
      if self._use_batched_updates:
        self._accumulated_error += error
        if self._update_i % self._batch_size == 0:
          self._optimise(self._accumulated_error / self._batch_size)
          self._accumulated_error = U.to_tensor(0.0, device=self._device)
      else:
        self._optimise(error)

  def rollout(self,
              initial_state,
              environment,
              render: bool = False,
              stat_writer=None,
              train: bool = True,
              max_length: int = None,
              disable_stdout=False,
              **kwargs):
    '''Perform a single rollout until termination in environment

    :param disable_stdout:
    :param stat_writer:
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
      self._update_i += 1

    episode_signal = []
    episode_length = 0
    episode_entropy = []
    episode_probs = []

    if isinstance(initial_state, EnvironmentSnapshot):
      state = initial_state.observables
    else:
      state = initial_state

    with draugr.scroll_plot_class(self._distribution_parameter_regressor.output_shape,
                                  render=render,
                                  window_length=66) as s:
      for t in tqdm(count(1), f'Update #{self._update_i}', leave=False, disable=disable_stdout):
        action, action_log_prob, entropy, probs = self.sample_action(state, render=render)

        state, signal, terminated, *_ = environment.step(action)

        if self._signal_clipping:
          signal = np.clip(signal,
                           self._signal_clip_low,
                           self._signal_clip_high)

        episode_signal.append(signal)
        episode_entropy.append(entropy.to('cpu').numpy())
        episode_probs.append(probs.detach().to('cpu').numpy())
        if train:
          self._trajectory_trace.add_point(signal, action_log_prob, entropy)

        if render:
          environment.render()
          s.draw(to_one_hot(self._distribution_parameter_regressor.output_shape, action)[0])

        if np.array(terminated).all() or (max_length and t > max_length):
          episode_length = t
          break

    if train:
      self.update_models()

    ep = np.array(episode_signal).sum(axis=0).mean()
    el = episode_length
    ee = np.array(episode_entropy).mean(axis=0).mean()
    pr = np.array(episode_probs)
    pr_mean = pr.mean(axis=0)[0]
    pr_std = pr.std(axis=0)[0]

    if stat_writer:
      stat_writer.scalar('duration', el, self._update_i)
      stat_writer.scalar('signal', ep, self._update_i)
      stat_writer.scalar('entropy', ee, self._update_i)
      stat_writer.bar('probs', pr_mean, self._update_i, yerr=pr_std)

    return ep, el, ee, pr_mean, pr_std

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

  # endregion


# region Test
def pg_test(rollouts=None, skip=True):
  import agent.configs.agent_test_configs.pg_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  train_agent(PGAgent,
              C,
              parse_args=False,
              training_session=parallelised_training,
              skip_confirmation=skip)


def pg_run(rollouts=None, skip=True):
  import agent.configs.agent_test_configs.pg_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  C.CONNECT_TO_RUNNING = True
  C.ENVIRONMENT_NAME = ""

  train_agent(PGAgent,
              C,
              training_session=parallelised_training(training_procedure=train_episodically),
              skip_confirmation=skip)


if __name__ == '__main__':
  # pg_test()
  pg_run()
# endregion
