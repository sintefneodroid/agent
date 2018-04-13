#!/usr/bin/env python3
# coding=utf-8
import math
import random
import time
from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

import utilities as U
from agents.value_agent import ValueAgent
from utilities.visualisation.term_plot import term_plot


class DQNAgent(ValueAgent):
  """

  """

  def __init__(self, config):
    super().__init__()
    self._step_n = 0
    self._rollout_i = 0

    self._memory = U.ReplayBuffer(config.REPLAY_MEMORY_SIZE)
    # self._memory = U.PrioritisedReplayMemory(config.REPLAY_MEMORY_SIZE)  # Cuda trouble

    self._use_cuda = config.USE_CUDA_IF_AVAILABLE if hasattr(config, 'USE_CUDA_IF_AVAILABLE') else False

    self._evaluation_function = config.EVALUATION_FUNCTION if hasattr(config,
                                                                      'EVALUATION_FUNCTION') else \
      F.smooth_l1_loss

    self._value_arch = config.VALUE_ARCH
    self._value_arch_parameters = config.VALUE_ARCH_PARAMS
    self._input_size = config.VALUE_ARCH_PARAMS['input_size']
    self._output_size = config.VALUE_ARCH_PARAMS['output_size']
    self._batch_size = config.BATCH_SIZE

    self._discount_factor = config.DISCOUNT_FACTOR
    self._learning_frequency = config.LEARNING_FREQUENCY
    self._initial_observation_period = config.INITIAL_OBSERVATION_PERIOD
    self._sync_target_model_frequency = config.SYNC_TARGET_MODEL_FREQUENCY

    self._state_tensor_type = config.STATE_TENSOR_TYPE
    self._value_type = config.VALUE_TENSOR_TYPE

    self._use_double_dqn = config.DOUBLE_DQN
    self._clamp_gradient = config.CLAMP_GRADIENT
    self._signal_clipping = config.SIGNAL_CLIPPING

    self._eps_start = config.EXPLORATION_EPSILON_START
    self._eps_end = config.EXPLORATION_EPSILON_END
    self._eps_decay = config.EXPLORATION_EPSILON_DECAY

    self._early_stopping_condition = None
    self._model = None
    self._target_model = None

    self._optimiser_type = config.OPTIMISER_TYPE if hasattr(config, 'OPTIMISER_TYPE') else torch.optim.Adam
    self._optimiser = None
    self._optimiser_alpha = config.OPTIMISER_ALPHA
    self._optimiser_learning_rate = config.OPTIMISER_LEARNING_RATE
    self._optimiser_epsilon = config.OPTIMISER_EPSILON
    self._optimiser_momentum = config.OPTIMISER_MOMENTUM

    self._end_training = False

  def __int__(self, **kwargs):
    for k, v in kwargs.items():
      self.__setattr__(f'_{str.lower(k)}', v)

  def stop_training(self):
    self._end_training = True

  def build_model(self, env):
    if type(self._input_size) is str:
      self._input_size = env.observation_space.shape
    print('observation dimensions: ', self._input_size)

    if type(self._output_size) is str:
      self._output_size = [env.action_space.num_binary_actions]
      if len(env.action_space.shape) > 1:
        self._output_size = env.action_space.shape
      else:
        self._output_size = [env.action_space.n]
    print('action dimensions: ', self._output_size)

    self._model, self._target_model, self._optimiser = self.__build_models__()

  def __build_models__(self):

    model = self._value_arch(**self._value_arch_parameters)

    target_model = self._value_arch(**self._value_arch_parameters)
    target_model.load_state_dict(model.state_dict())

    if self._use_cuda:
      model = model.cuda()
      target_model = target_model.cuda()

    optimiser = self._optimiser_type(model.parameters(),
                                     lr=self._optimiser_learning_rate,
                                     eps=self._optimiser_epsilon,
                                     alpha=self._optimiser_alpha,
                                     momentum=self._optimiser_momentum)

    return model, target_model, optimiser

  def optimise_wrt(self, error):
    """

    :param error:
    :type error:
    :return:
    """
    self._optimiser.zero_grad()
    error.backward()
    if self._clamp_gradient:
      for params in self._model.parameters():
        params.grad.data.clamp_(-1, 1)
    self._optimiser.step()

  def evaluate_td_loss(self, batch):
    """

    :param batch:
    :type batch:
    :return:
    :rtype:
    """

    # Torchify batch
    states = U.to_var(batch.state, use_cuda=self._use_cuda).view(-1, self._input_size[0])
    action_indices = U.to_var(batch.action, 'long', use_cuda=self._use_cuda).view(-1, 1)
    true_signals = U.to_var(batch.signal, use_cuda=self._use_cuda)
    non_terminal_mask = U.to_tensor(batch.non_terminal, 'byte', use_cuda=self._use_cuda)
    non_terminal_successors = U.to_tensor([states
                                           for (states, non_terminal_mask)
                                           in zip(batch.successor_state, batch.non_terminal)
                                           if non_terminal_mask], 'float', use_cuda=self._use_cuda)
    if not len(non_terminal_successors) > 0:
      return 0  # Nothing to be learned, all states are terminal
    non_terminal_successors_var = U.to_var(non_terminal_successors, use_cuda=self._use_cuda, volatile=True)

    # Calculate Q of successors
    Q_successors = self._model(non_terminal_successors_var)
    Q_successors_max_action_indices = Q_successors.max(1)[1].view(-1, 1)
    if self._use_double_dqn:
      Q_successors = self._target_model(non_terminal_successors_var)
    Q_max_successor = Variable(torch.zeros(self._batch_size).type(self._value_type))
    Q_max_successor[non_terminal_mask] = Q_successors.gather(1, Q_successors_max_action_indices)

    # Integrate with the true signal
    Q_expected = true_signals + (self._discount_factor * Q_max_successor)

    # Calculate Q of state
    Q_state = self._model(states).gather(1, action_indices)

    return self._evaluation_function(Q_state, Q_expected)

  def rollout(self, initial_state, environment, render=False):
    self._rollout_i += 1

    state = initial_state
    episode_signal = 0
    episode_length = 0
    episode_td_loss = 0

    T = count(1)
    T = tqdm(T, f'Rollout #{self._rollout_i}', leave=False)

    for t in T:
      self._step_n += 1

      action = self.sample_action(state)
      next_state, signal, terminated, info = environment.step(action)

      if render:
        environment.render()

      if self._signal_clipping:
        signal = np.clip(signal, -1.0, 1.0)

      successor_state = None
      if not terminated:  # If environment terminated then there is no successor state
        successor_state = next_state

      self._memory.add_transition(state, action, signal, successor_state, not terminated)
      state = next_state

      td_l = 0

      if len(self._memory) >= self._batch_size and \
          self._step_n > self._initial_observation_period and \
          self._step_n % self._learning_frequency == 0:

        # indices, transitions = self._memory.sample_transitions(self.C.BATCH_SIZE)
        transitions = self._memory.sample_transitions(self._batch_size)

        td_loss = self.evaluate_td_loss(transitions)
        self.optimise_wrt(td_loss)

        td_l = td_loss.data[0]
        # self._memory.batch_update(indices, errors.tolist())  # Cuda trouble

        T.set_description(f'TD loss: {td_l}')

      if self._use_double_dqn and self._step_n % self._sync_target_model_frequency == 0:
        self._target_model.load_state_dict(self._model.state_dict())
        T.write('Target Model Synced')

      episode_signal += signal
      episode_td_loss += td_l

      if terminated:
        episode_length = t
        break

    return episode_signal, episode_length, episode_td_loss

  def forward(self, state, *args, **kwargs):
    model_input = Variable(state, volatile=True).type(self._state_tensor_type)
    return self._model(model_input)

  def sample_action(self, state):
    """

    :param state:
    :return:
    """
    if self.epsilon_random(self._step_n) and self._step_n > self._initial_observation_period:
      return self.sample_model(state)
    return self.sample_random_process()

  def sample_model(self, state):
    model_input = U.to_var([state], volatile=True, use_cuda=self._use_cuda)
    action_value_estimates = self._model(model_input)
    max_value_action_idx = action_value_estimates.max(1)[1].data[0]
    # max_value_action_idx = np.argmax(action_value_estimates.data.cpu().numpy()[0])
    return max_value_action_idx

  def sample_random_process(self):
    sample = np.random.choice(self._output_size[0])
    return sample

  def epsilon_random(self, steps_taken):
    """
    :param steps_taken:
    :return:
    """
    # assert type(steps_taken) is int

    if steps_taken == 0:
      return True

    sample = random.random()

    a = self._eps_start - self._eps_end
    b = math.exp(-1. * steps_taken / self._eps_decay)
    eps_threshold = self._eps_end + a * b

    return sample > eps_threshold

  def step(self, state, env):
    self._step_n += 1
    a = self.sample_action(state)
    return env.step(a)

  def infer(self, environment, render=True):
    for episode_i in count(1):
      print('Episode {}'.format(episode_i))

      state = environment.reset()
      for episode_frame_i in count(1):

        a = self.sample_model(state)
        state, reward, terminated, info = environment.step(a)
        if render:
          environment.render()

        if terminated:
          break

  def save_model(self, C):
    U.save_model(self._model, C)

  def load_model(self, model_path):
    print('Loading latest model: ' + model_path)
    self._model = self._value_arch(**self._value_arch_parameters)
    self._model.load_state_dict(torch.load(model_path))
    # self._model = self._model.eval()
    # self._model.train(False)
    if self._use_cuda:
      self._model = self._model.cuda()
    else:
      self._model = self._model.cpu()

  def train_episodic(self,
                     _environment,
                     rollouts=1000,
                     render=False,
                     render_frequency=400,
                     stat_frequency=400):
    """

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
    """
    running_signal = 0
    dur = 0
    td_loss = 0
    running_signals = []
    durations = []
    td_losses = []

    E = range(1, rollouts)
    E = tqdm(E, leave=True)

    training_start_timestamp = time.time()

    for episode_i in E:
      initial_state = _environment.reset()

      if episode_i % stat_frequency == 0:
        t_episode = [i for i in range(1, episode_i + 1)]
        term_plot(t_episode,
                  running_signals,
                  'Running Signal',
                  printer=E.write,
                  percent_size=(1, .24))
        term_plot(t_episode,
                  durations,
                  'Duration',
                  printer=E.write, percent_size=(1,
                                                 .24))
        term_plot(t_episode,
                  td_losses,
                  'TD Loss',
                  printer=E.write,
                  percent_size=(1, .24))
        E.set_description(f'Episode: {episode_i}, '
                          f'Running Signal: {running_signal}, '
                          f'Duration: {dur}, '
                          f'TD Loss: {td_loss}')

      if render and episode_i % render_frequency == 0:
        signal, dur, *stats = self.rollout(initial_state, _environment, render=render)
      else:
        signal, dur, *stats = self.rollout(initial_state, _environment)

      running_signal = running_signal * 0.99 + signal * 0.01
      running_signals.append(running_signal)
      durations.append(dur)
      td_loss = stats[0]
      td_losses.append(td_loss)

      if self._end_training:
        break

    time_elapsed = time.time() - training_start_timestamp
    end_message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed %60:.0f}s'
    print('\n{} {} {}\n'.format('-' * 9, end_message, '-' * 9))

    return self._model, []


def test_dqn_agent(config):
  environment = gym.make(config.ENVIRONMENT_NAME)
  environment.seed(config.RANDOM_SEED)

  config.VALUE_ARCH_PARAMS['input_size'] = [4]
  config.VALUE_ARCH_PARAMS['output_size'] = [environment.action_space.n]

  agent = DQNAgent(C)
  listener = U.add_early_stopping_key_combination(agent.stop_training)

  _trained_model=None
  listener.start()
  try:
    agent.build_model(environment)
    _trained_model, training_statistics, *_ = agent.train_episodic(environment, config.EPISODES,
                                                                   render=config.RENDER_ENVIRONMENT)
  finally:
    listener.stop()

  U.save_model(_trained_model, C)

  environment.close()

def main2():
  args = parser.parse_args()
  torch.manual_seed(args.seed)

  if not os.path.exists(args.dump_location):
    os.makedirs(args.dump_location)

  logging.basicConfig(
      filename=args.dump_location +
               'train.log',
      level=logging.INFO)

  assert args.evaluate == 0 or args.num_processes == 0, \
    "Can't train while evaluating, either n=0 or e=0"

  class Net(torch.nn.Module):
    def __init__(self, args):
      super(Net, self).__init__()
      self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
      self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
      self.conv2_drop = torch.nn.Dropout2d()
      self.fc1 = torch.nn.Linear(320, 50)
      self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
      x = F.relu(F.max_pool2d(self.conv1(x), 2))
      x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
      x = x.view(-1, 320)
      x = F.relu(self.fc1(x))
      x = F.dropout(x, training=self.training)
      x = self.fc2(x)
      return F.log_softmax(x, dim=1)

  def train(rank, args, model):
    torch.manual_seed(args.seed + rank)

    pass

  def test(rank, args, model):
    torch.manual_seed(args.seed + rank)

    pass

  shared_model = Net(args)

  if args.load != "0":
    shared_model.load_state_dict(torch.load(args.load))
  shared_model.share_memory()

  signal.signal(signal.SIGINT, signal.signal(signal.SIGINT, signal.SIG_IGN))
  processes = []

  p = TMP.Process(target=test, args=(args.num_processes, args, shared_model))
  p.start()
  processes.append(p)

  for rank in range(0, args.num_processes):
    p = TMP.Process(target=train, args=(rank, args, shared_model))
    p.start()
    processes.append(p)

  try:
    for p in processes:
      p.join()
  except KeyboardInterrupt:
    print("Stopping training. " +
          "Best model stored at {}model_best".format(args.dump_location))
    for p in processes:
      p.terminate()

if __name__ == '__main__':
  import gym
  import configs.dqn_config as C
  import argparse
  import logging
  import os
  import signal
  import torch
  import torch.multiprocessing as TMP

  parser = argparse.ArgumentParser(description='DQN Agent')
  # parser.add_argument('integers', metavar='N', type=int, nargs='+', help='')
  parser.add_argument('--ENVIRONMENT_NAME', '-E', type=str, default=C.ENVIRONMENT_NAME,
                      metavar='ENVIRONMENT_NAME',
                      help='name of the environment to run')
  parser.add_argument('--PRETRAINED_PATH', '-T', metavar='PATH', type=str, default='',
                      help='path of pre-trained model')
  parser.add_argument('--RENDER_ENVIRONMENT', '-R', action='store_true',
                      default=C.RENDER_ENVIRONMENT,
                      help='render the environment')
  parser.add_argument('--NUM_WORKERS', '-N', type=int, default=4, metavar='NUM_WORKERS',
                      help='number of threads for agent (default: 4)')
  parser.add_argument('--RANDOM_SEED', '-S', type=int, default=1, metavar='RANDOM_SEED',
                      help='random seed (default: 1)')
  args = parser.parse_args()

  for k, arg in args.__dict__.items():
    setattr(C, k, arg)
    print(k, arg)

  try:
    test_dqn_agent(C)
  except KeyboardInterrupt:
    print('Stopping')

