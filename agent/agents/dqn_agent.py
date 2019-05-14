#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
from warg import NOD, NamedOrderedDictionary

from agent.architectures import MLP
from agent.procedures.train_agent import agent_test_main, parallel_train_agent_procedure
from agent.utilities import get_screen
from agent.utilities.specifications.generalised_delayed_construction_specification import GDCS
from agent.utilities.visualisation.experimental.statistics_plot import plot_durations

__author__ = 'cnheider'
from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from agent import utilities as U
from agent.agents.abstract.value_agent import ValueAgent


class DQNAgent(ValueAgent):
  '''

'''

  # region Protected

  def __defaults__(self) -> None:
    self._memory_buffer = U.ReplayBuffer(10000)
    # self._memory = U.PrioritisedReplayMemory(config.REPLAY_MEMORY_SIZE)  # Cuda trouble

    self._use_cuda = False

    self._evaluation_function = F.smooth_l1_loss

    self._value_arch_spec = GDCS(MLP, NamedOrderedDictionary({
      'input_size':             None,  # Obtain from environment
      'hidden_layers':          None,
      'output_size':            None,  # Obtain from environment
      'hidden_layer_activation':torch.tanh,
      'use_bias':               True,
      }))

    self._batch_size = 128

    self._discount_factor = 0.99
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

    self._optimiser = self._optimiser_spec.constructor(
        self._value_model.parameters(),
        **self._optimiser_spec.kwargs
        )

  def _optimise_wrt(self, error, **kwargs):
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

  def evaluate(self, batch, *args, **kwargs):
    '''

:param batch:
:type batch:
:return:
:rtype:
'''
    states = U.to_tensor(batch.state,
                         dtype=self._state_type,
                         device=self._device).view(-1, *self._input_size)

    true_signals = U.to_tensor(batch.signal,
                               dtype=self._value_type,
                               device=self._device).view(-1, 1)

    action_indices = U.to_tensor(batch.action,
                                 dtype=self._action_type,
                                 device=self._device).view(-1, 1)

    non_terminal_mask = U.to_tensor(batch.non_terminal,
                                    dtype=torch.float,
                                    device=self._device).view(-1, 1)

    successor_states = U.to_tensor(batch.successor_state,
                                   dtype=self._state_type,
                                   device=self._device).view(-1, *self._input_size)

    # Calculate Q of successors
    with torch.no_grad():
      Q_successors = self._value_model(successor_states)

    Q_successors_max_action_indices = Q_successors.max(-1)[1]
    Q_successors_max_action_indices = Q_successors_max_action_indices.view(-1, 1)
    if self._use_double_dqn:
      with torch.no_grad():
        Q_successors = self._target_value_model(successor_states)

    max_next_values = Q_successors.gather(1, Q_successors_max_action_indices)
    # a = Q_max_successor[non_terminal_mask]
    Q_max_successor = max_next_values * non_terminal_mask

    # Integrate with the true signal
    Q_expected = true_signals + (self._discount_factor * Q_max_successor).view(-1, 1)

    # Calculate Q of state
    Q_state = self._value_model(states).gather(1, action_indices)

    return self._evaluation_function(Q_state, Q_expected)

  def update(self):
    error = 0
    if self._batch_size < len(self._memory_buffer):
      # indices, transitions = self._memory.sample_transitions(self.C.BATCH_SIZE)
      transitions = self._memory_buffer.sample_transitions(self._batch_size)

      td_error = self.evaluate(transitions)
      self._optimise_wrt(td_error)

      error = td_error.item()
      # self._memory.batch_update(indices, errors.tolist())  # Cuda trouble

    return error

  def rollout(self, initial_state, environment, render=False, train=True, random_sample=True, **kwargs):
    self._rollout_i += 1

    state = np.array(initial_state)
    episode_signal = []
    episode_length = []
    episode_td_error = []

    T = count(1)
    T = tqdm(T, f'Rollout #{self._rollout_i}', leave=False)

    for t in T:
      self._step_i += 1

      action = self.sample_action(state, random_sample=random_sample)
      next_state, signal, terminated, *_ = environment.react(action)

      if render:
        environment.render()

      if self._signal_clipping:
        signal = np.clip(signal, -1.0, 1.0)

      successor_state = np.array(next_state)

      terminated = np.array(terminated)

      self._memory_buffer.add_transition(state,
                                         action,
                                         signal,
                                         successor_state,
                                         [not t for t in terminated]
                                         )

      td_error = 0

      if (len(self._memory_buffer) >= self._batch_size
          and self._step_i > self._initial_observation_period
          and self._step_i % self._learning_frequency == 0
      ):

        td_error = self.update()

        # T.set_description(f'TD error: {td_error}')

      if (self._use_double_dqn
          and self._step_i % self._sync_target_model_frequency == 0
      ):
        self._target_value_model = U.copy_state(target=self._target_value_model, source=self._value_model)
        if self._verbose:
          T.write('Target Model Synced')

      episode_signal.append(signal)
      episode_td_error.append(td_error)

      if terminated.all():
        episode_length = t
        break

      state = next_state

    ep = np.array(episode_signal).mean()
    el = episode_length
    ee = np.array(episode_td_error).mean()
    return ep, el, ee

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
def test_cnn_dqn_agent(config):
  import gym

  env = gym.make(config.ENVIRONMENT_NAME).unwrapped
  env.seed(config.SEED)

  is_ipython = 'inline' in matplotlib.get_backend()
  if is_ipython:
    pass

  plt.ion()

  episode_durations = []

  agent = DQNAgent(config)
  agent.build(env)

  episodes = tqdm(range(config.ROLLOUTS), leave=False)
  for episode_i in episodes:
    episodes.set_description(f'Episode:{episode_i}')
    env.reset()
    last_screen = U.transform_screen(get_screen(env), agent.device)
    current_screen = U.transform_screen(get_screen(env), agent.device)
    state = current_screen - last_screen

    rollout = tqdm(count(), leave=False)
    for t in rollout:

      action, (_, signal, terminated, *_) = agent.step(state, env)

      last_screen = current_screen
      current_screen = U.transform_screen(get_screen(env), agent.device)

      successor_state = None
      if not terminated:
        successor_state = current_screen - last_screen

      if agent._signal_clipping:
        signal = np.clip(signal, -1.0, 1.0)

      agent._memory_buffer.add_transition(state, action, signal, successor_state, not terminated)

      agent.update()
      if terminated:
        episode_durations.append(t + 1)
        plot_durations(episode_durations=episode_durations)
        break

      state = successor_state

  env.render()
  env.close()
  plt.ioff()
  plt.show()


def dqn_test(rollouts=None):
  import agent.configs.agent_test_configs.dqn_test_config as C

  # import configs.cnn_dqn_config as C
  if rollouts:
    C.ROLLOUTS = rollouts

  agent_test_main(DQNAgent, C, parse_args=False, training_procedure=parallel_train_agent_procedure)
  # test_cnn_dqn_agent(C)


if __name__ == '__main__':
  dqn_test()
# endregion
