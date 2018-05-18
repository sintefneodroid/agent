#!/usr/bin/env python3
# coding=utf-8
import matplotlib

from utilities.transformation.extraction import get_screen
from utilities.visualisation.statistics_plot import plot_durations

__author__ = 'cnheider'
import time
from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pylab as plt

import utilities as U
from agents.value_agent import ValueAgent
from utilities.visualisation.term_plot import term_plot


# noinspection PyCallingNonCallable
class DQNAgent(ValueAgent):
  '''

'''

  def __local_defaults__(self):
    self._memory = U.ReplayBuffer(10000)
    # self._memory = U.PrioritisedReplayMemory(config.REPLAY_MEMORY_SIZE)  # Cuda trouble

    self._use_cuda = False

    self._evaluation_function = F.smooth_l1_loss

    self._value_arch = U.MLP
    self._value_arch_parameters = {
      'input_size': None,  # Obtain from environment
      'hidden_size':[64, 32, 16],
      'output_size':None,  # Obtain from environment
      'activation': F.relu,
      'use_bias':   True,
      }

    self._batch_size = 128

    self._discount_factor = 0.99
    self._learning_frequency = 1
    self._initial_observation_period = 0
    self._sync_target_model_frequency = 1000

    self._state_tensor_type = torch.float
    self._value_tensor_type = torch.float
    self._action_tensor_type = torch.long

    self._use_double_dqn = True
    self._clamp_gradient = False
    self._signal_clipping = True

    self._eps_start = 1.0
    self._eps_end = 0.02
    self._eps_decay = 400

    self._early_stopping_condition = None
    self._target_value_model = None

    self._optimiser_type = torch.optim.RMSprop
    self._optimiser = None
    self._optimiser_alpha = 0.9
    self._optimiser_learning_rate = 0.0025
    self._optimiser_epsilon = 1e-02
    self._optimiser_momentum = 0.0

  def __build_models__(self):
    self._value_arch_parameters['input_size'] = self._input_size
    self._value_arch_parameters['output_size'] = self._output_size

    value_model = self._value_arch(**self._value_arch_parameters).to(self._device)

    target_value_model = self._value_arch(**self._value_arch_parameters).to(
        self._device
        )
    target_value_model.load_state_dict(value_model.state_dict())
    target_value_model.eval()

    optimiser = self._optimiser_type(
        value_model.parameters(),
        lr=self._optimiser_learning_rate,
        eps=self._optimiser_epsilon,
        # alpha=self._optimiser_alpha,
        # momentum=self._optimiser_momentum,
        )

    self._value_model, self._target_value_model, self._optimiser = value_model, target_value_model, optimiser

  def __optimise_wrt__(self, error, **kwargs):
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

  def evaluate(self, batch, *args, **kwargs):
    '''

:param batch:
:type batch:
:return:
:rtype:
'''

    # Torchify batch
    # states = torch.tensor(
    #    batch.state, device=self._device, dtype=self._state_tensor_type
    #    ).view(
    #    -1, *self._input_size
    #    )
    if type(batch.state[0]) is not torch.Tensor:
      states = torch.tensor(batch.state, dtype=self._state_tensor_type, device=self._device).view(
          -1, *self._input_size
          )
    else:
      states = torch.cat(batch.state).to(self._device)

    action_indices = torch.tensor(
        batch.action, dtype=self._action_tensor_type, device=self._device
        ).view(
        -1, 1
        )
    true_signals = torch.tensor(batch.signal, dtype=self._value_tensor_type, device=self._device).view(-1, 1)


    non_terminal_mask = torch.tensor(
        batch.non_terminal, dtype=torch.uint8, device=self._device
        )
    nts = [ state for (state, non_terminal_mask) in zip( batch.successor_state, batch.non_terminal ) if non_terminal_mask ]
    if type(nts[0]) is not torch.Tensor:
      non_terminal_successors = torch.tensor(nts, dtype=self._state_tensor_type, device=self._device).view(
          -1, *self._input_size
          )

    else:
      non_terminal_successors = torch.cat(nts).to(self._device)

    if not len(non_terminal_successors) > 0:
      return 0  # Nothing to be learned, all states are terminal

    # Calculate Q of successors
    with torch.no_grad():
      Q_successors = self._value_model(non_terminal_successors)
    Q_successors_max_action_indices = Q_successors.max(1)[1].view(-1, 1)
    if self._use_double_dqn:
      with torch.no_grad():
        Q_successors = self._target_value_model(non_terminal_successors)
    Q_max_successor = torch.zeros(
        self._batch_size, dtype=self._value_tensor_type, device=self._device
        )
    Q_max_successor[non_terminal_mask] = Q_successors.gather(
        1, Q_successors_max_action_indices
        ).squeeze()

    # Integrate with the true signal
    Q_expected = true_signals + (self._discount_factor * Q_max_successor).view(
        -1, 1
        )

    # Calculate Q of state
    Q_state = self._value_model(states).gather(1, action_indices)

    return self._evaluation_function(Q_state, Q_expected)

  def update_models(self):
    error = 0
    if self._batch_size < len(self._memory):
      # indices, transitions = self._memory.sample_transitions(self.C.BATCH_SIZE)
      transitions = self._memory.sample_transitions(self._batch_size)

      td_error = self.evaluate(transitions)
      self.__optimise_wrt__(td_error)

      error = td_error.item()
      # self._memory.batch_update(indices, errors.tolist())  # Cuda trouble

    return error

  def rollout(self, initial_state, environment, render=False, **kwargs):
    self._rollout_i += 1

    state = initial_state
    episode_signal = 0
    episode_length = 0
    episode_td_error = 0

    T = count(1)
    T = tqdm(T, f'Rollout #{self._rollout_i}', leave=False)

    for t in T:
      self._step_i += 1

      action = self.sample_action(state)
      next_state, signal, terminated, info = environment.step(action)

      if render:
        environment.render()

      if self._signal_clipping:
        signal = np.clip(signal, -1.0, 1.0)

      successor_state = None
      if not terminated:  # If environment terminated then there is no successor state
        successor_state = next_state

      self._memory.add_transition(
          state, action, signal, successor_state, not terminated
          )

      td_error = 0

      if (
          len(self._memory) >= self._batch_size
          and self._step_i > self._initial_observation_period
          and self._step_i % self._learning_frequency == 0
      ):

        td_error = self.update_models()

        T.set_description(f'TD error: {td_error}')

      if (
          self._use_double_dqn
          and self._step_i % self._sync_target_model_frequency == 0
      ):
        self._target_value_model.load_state_dict(self._value_model.state_dict())
        if self._verbose:
          T.write('Target Model Synced')

      episode_signal += signal
      episode_td_error += td_error

      if terminated:
        episode_length = t
        break

      state = next_state

    return episode_signal, episode_length, episode_td_error

  def infer(self, state, **kwargs):
    if type(state) is not torch.Tensor:
      model_input = torch.tensor(
          [state], device=self._device, dtype=self._state_tensor_type
          )
    else:
      model_input = state
    with torch.no_grad():
      value = self._value_model(model_input)
    return value

  def __sample_model__(self, state, **kwargs):
    if type(state) is not torch.Tensor:
      model_input = torch.tensor(
          [state], device=self._device, dtype=self._state_tensor_type
          )
    else:
      model_input = state

    with torch.no_grad():
      action_value_estimates = self._value_model(model_input)
    max_value_action_idx = action_value_estimates.max(1)[1].item()
    return max_value_action_idx

  def step(self, state, env):
    action = self.sample_action(state)
    return action, env.step(action)

  def train_episodic(
      self,
      _environment,
      rollouts=1000,
      render=False,
      render_frequency=400,
      stat_frequency=400,
      ):
    '''

:param _environment:
:type _environment:
:param rollouts:
:type rollouts:
:param render:
:type render:
:param render_frequency:
:type render_frequency:
:param stat_frequency:
:type stat_frequency:
:return:
:rtype:
'''
    running_signal = 0
    dur = 0
    td_error = 0
    running_signals = []
    durations = []
    td_errors = []

    E = range(1, rollouts)
    E = tqdm(E, leave=False)

    training_start_timestamp = time.time()

    for episode_i in E:
      initial_state = _environment.reset()

      if episode_i % stat_frequency == 0:
        t_episode = [i for i in range(1, episode_i + 1)]
        term_plot(
            t_episode,
            running_signals,
            'Running Signal',
            printer=E.write,
            percent_size=(1, .24),
            )
        term_plot(
            t_episode,
            durations,
            'Duration',
            printer=E.write,
            percent_size=(1, .24),
            )
        term_plot(
            t_episode,
            td_errors,
            'TD Error',
            printer=E.write,
            percent_size=(1, .24),
            )
        E.set_description(
            f'Episode: {episode_i}, '
            f'Running Signal: {running_signal}, '
            f'Duration: {dur}, '
            f'TD Error: {td_error}'
            )

      if render and episode_i % render_frequency == 0:
        signal, dur, *stats = self.rollout(
            initial_state, _environment, render=render
            )
      else:
        signal, dur, *stats = self.rollout(initial_state, _environment)

      running_signal = running_signal * 0.99 + signal * 0.01
      running_signals.append(running_signal)
      durations.append(dur)
      td_error = stats[0]
      td_errors.append(td_error)

      if self._end_training:
        break

    time_elapsed = time.time() - training_start_timestamp
    end_message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed %60:.0f}s'
    print('\n{} {} {}\n'.format('-' * 9, end_message, '-' * 9))

    return self._value_model, []


def test_dqn_agent(config):
  import gym

  device = torch.device('cuda' if config.USE_CUDA else 'cpu')

  environment = gym.make(config.ENVIRONMENT_NAME)
  environment.seed(config.SEED)

  agent = DQNAgent(config)
  agent.build_agent(environment, device)

  listener = U.add_early_stopping_key_combination(agent.stop_training)

  listener.start()
  try:

    trained_model, training_statistics, *_ = agent.train_episodic(
        environment, config.ROLLOUTS, render=config.RENDER_ENVIRONMENT
        )
  finally:
    listener.stop()

  U.save_model(trained_model, config)

  environment.close()


def test_cnn_dqn_agent(config):
  import gym

  device = torch.device('cuda' if config.USE_CUDA else 'cpu')

  env = gym.make(config.ENVIRONMENT_NAME).unwrapped
  env.seed(config.SEED)

  is_ipython = 'inline' in matplotlib.get_backend()
  if is_ipython:
    pass

  plt.ion()

  episode_durations = []

  agent = DQNAgent(C)
  agent.build_agent(env, device)

  episodes = tqdm(range(C.ROLLOUTS), leave=False)
  for episode_i in episodes:
    episodes.set_description(f'Episode:{episode_i}')
    env.reset()
    last_screen = U.transform_screen(get_screen(env), device)
    current_screen = U.transform_screen(get_screen(env), device)
    state = current_screen - last_screen

    rollout = tqdm(count(), leave=False)
    for t in rollout:

      action, (_, signal, terminated, *_) = agent.step(state, env)

      last_screen = current_screen
      current_screen = U.transform_screen(get_screen(env), device)

      successor_state = None
      if not terminated:
        successor_state = current_screen - last_screen

      if agent._signal_clipping:
        signal = np.clip(signal, -1.0, 1.0)

      agent._memory.add_transition(state, action, signal, successor_state, not terminated)

      agent.update_models()
      if terminated:
        episode_durations.append(t + 1)
        plot_durations(episode_durations=episode_durations)
        break

      state = successor_state

  env.render()
  env.close()
  plt.ioff()
  plt.show()


if __name__ == '__main__':
  import configs.dqn_config as C
  # import configs.cnn_dqn_config as C

  from configs.arguments import parse_arguments

  args = parse_arguments('DQN agent', C)

  for k, arg in args.__dict__.items():
    setattr(C, k, arg)

  print(f'Using config: {C}')
  if not args.skip_confirmation:
    for k, arg in U.get_upper_vars_of(C).items():
      print(f'{k} = {arg}')
    input('\nPress any key to begin... ')

  try:
    test_dqn_agent(C)
    # test_cnn_dqn_agent(C)
  except KeyboardInterrupt:
    print('Stopping')

  torch.cuda.empty_cache()
