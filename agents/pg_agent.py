#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'

import time
from itertools import count

import gym
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from utilities.visualisation.term_plot import term_plot

tqdm.monitor_interval = 0

import utilities as U

from torch.distributions import Categorical
from agents.policy_agent import PolicyAgent


class PGAgent(PolicyAgent):

  def build_model(self, env):
    if type(self._policy_arch_params['input_size']) == str:
      self._policy_arch_params['input_size'] = env.observation_space.shape
    print('observation dimensions: ', self._policy_arch_params['input_size'])

    if type(self._policy_arch_params['output_size']) == str:
      if hasattr(env.action_space, 'num_binary_actions'):
        self._policy_arch_params['output_size'] = [env.action_space.num_binary_actions]
      else:
        self._policy_arch_params['output_size'] = [env.action_space.n]
    print('action dimensions: ', self._policy_arch_params['output_size'])

    self._model = self.__build_model__()

  def __build_model__(self):

    model = self._policy_arch(**self._policy_arch_params)

    if self._use_cuda_if_available:
      model = model.cuda()

    self.optimiser = self._optimiser_type(model.parameters(), lr=self._optimiser_learning_rate,
                                          weight_decay=self._optimiser_weight_decay)

    return model

  def sample_action(self, state):
    state_var = U.to_var(state, use_cuda=self._use_cuda_if_available, unsqueeze=True)
    probs = self._model(state_var)
    m = Categorical(probs)
    action = m.sample()
    a = action.cpu().data.numpy()[0]
    # a = np.argmax(a)
    return a, m.log_prob(action), U.log_entropy(probs)

  def sample_cont_action(self, state):
    state_var = U.to_var(state, use_cuda=self._use_cuda_if_available, unsqueeze=True)
    mu, sigma_sq = self._model(state_var)  # requires MultiheadedMLP

    eps = torch.randn(mu.size())
    # calculate the probability
    action = (mu + sigma_sq.sqrt() * Variable(eps).cuda()).data
    prob = U.normal(action, mu, sigma_sq)
    entropy = -0.5 * ((sigma_sq + 2 * U.pi_torch(self._use_cuda_if_available).expand_as(sigma_sq)).log() + 1)

    log_prob = prob.log()
    return action, log_prob, entropy

  def evaluate(self, not_used):
    R = 0
    policy_loss = []
    signals = []
    for r in self._trajectory.signals[::-1]:
      R = r + self._pg_gamma * R
      signals.insert(0, R)

    signals = U.to_tensor(signals, use_cuda=self._use_cuda_if_available)

    stddev = signals.std()  # + np.finfo(np.float32).eps) for no zero division
    if signals.shape[0] > 1 and stddev > 0:
      signals = (signals - signals.mean()) / stddev
    else:
      return None

    for log_prob, signal, entropy in zip(self._trajectory.log_probs, signals, self._trajectory.entropies):
      policy_loss.append(-log_prob * signal - self._pg_entropy_reg * entropy)

    loss = torch.cat(policy_loss).sum()
    return loss

  def rollout(self, initial_state, environment, render=False):
    self._rollout_i += 1

    episode_signal = 0
    episode_length = 0
    episode_entropy = 0

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
      self._trajectory.remember(signal, action_log_probs, entropy)

      if render:
        environment.render()

      if terminated:
        episode_length = t
        break

    error = self.evaluate(1)
    self._trajectory.forget()
    if error is not None:
      if self._use_batched_updates:
        self._accumulated_error += error
        if self._rollout_i % self._batch_size == 0:
          self.optimise_wrt(self._accumulated_error / self._batch_size)
          self._accumulated_error = U.to_var([0.0], use_cuda=self._use_cuda_if_available)
      else:
        self.optimise_wrt(error)

    return episode_signal, episode_length, episode_entropy / episode_length

  def __init__(self, config=None, *args, **kwargs):

    self.C = config

    self._policy_arch = None
    self._policy_arch_params = None
    self._use_cuda_if_available = False
    self._pg_gamma = config.PG_GAMMA
    self._pg_learning_rate = config.PG_LR
    self._optimiser_type = config.OPTIMISER_TYPE
    self._optimiser_learning_rate = config.OPTIMISER_LEARNING_RATE

    self._pg_entropy_reg = config.PG_ENTROPY_REG
    self._signal_clipping = config.SIGNAL_CLIPPING

    self._optimiser_weight_decay = config.OPTIMISER_WEIGHT_DECAY
    self._evaluation_function = config.EVALUATION_FUNCTION()

    self._trajectory = U.Trajectory()
    self._use_batched_updates = False  # self.C.BATCHED_UPDATES
    self._batch_size = 5  # self.C.BATCH_SIZE
    self._accumulated_error = U.to_var([0.0], use_cuda=self._use_cuda_if_available)

    super().__init__(config, *args, **kwargs)

  def optimise_wrt(self, loss):
    self.optimiser.zero_grad()
    loss.backward()
    for params in self._model.parameters():
      params.grad.data.clamp_(-1, 1)
    self.optimiser.step()

  def save_model(self, C):
    U.save_model(self._model, C)

  def load_model(self, model_path, evaluation=False):
    print('loading latest model: ' + model_path)
    self._model = self._policy_arch(**self._policy_arch_params)
    self._model.load_state_dict(torch.load(model_path))
    if evaluation:
      self.model = self._model.eval()
      self._model.train(False)
    if self._use_cuda_if_available:
      self._model = self._model.cuda()

  def infer(self, env, render=True):

    for episode_i in count(1):
      print('Episode {}'.format(episode_i))
      state = env.reset()

      for episode_frame_i in count(1):

        action, *_ = self.sample_action(state)
        state, reward, terminated, info = env.step(action)
        if render:
          env.render()

        if terminated:
          break

  def train(self, _environment, rollouts=1000, render=False, render_frequency=100, stat_frequency=100):

    training_start_timestamp = time.time()
    E = range(1, rollouts)
    E = tqdm(E, f'Episode: {1}', leave=True)

    running_length = 0
    running_lengths = []
    for episode_i in E:
      initial_state = _environment.reset()

      if episode_i % stat_frequency == 0:
        x_t = [i for i in range(1, episode_i + 1)]
        term_plot(x_t, running_lengths, 'Running Lengths', printer=E.write)
        E.set_description(f'Episode: {episode_i}, Running length: {running_length}')

      if render and episode_i % render_frequency == 0:
        signal, dur, *stats = self.rollout(initial_state, _environment, render=render)
      else:
        signal, dur, *stats = self.rollout(initial_state, _environment)

      running_length = running_length * 0.99 + dur * 0.01
      running_lengths.append(running_length)

    time_elapsed = time.time() - training_start_timestamp
    end_message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed %60:.0f}s'
    print('\n{} {} {}\n'.format('-' * 9, end_message, '-' * 9))

    return self._model, []


def test_pg_agent(config):
  env_name = 'CartPole-v0'
  # env_name = 'LunarLander-v2' # (coord_x, coord_y, vel_x, vel_y, angle, angular_vel, l_leg_on_ground,
  # r_leg_on_ground)

  env = gym.make(env_name)
  env.seed(config.SEED)

  torch.manual_seed(config.SEED)
  config.USE_CUDA_IF_AVAILABLE = False
  config.ARCH_PARAMS['input_size'] = [4]
  config.ARCH_PARAMS['output_size'] = [env.action_space.n]

  agent = PGAgent(config)
  agent.build_model(env)

  _trained_model, training_statistics, *_ = agent.train(env, config.MAX_ROLLOUT_LENGTH,
                                                        render=config.RENDER_ENVIRONMENT)
  U.save_model(_trained_model, config)

  env.close()


#
# def test_agent():
#   import configs.pg_config as C
#
#   env_name = 'CartPole-v0' # 'LunarLander-v2'
#   env = gym.make(env_name)
#   env.seed(C.SEED)
#
#   torch.manual_seed(C.SEED)
#   C.USE_CUDA_IF_AVAILABLE = False
#   C.ARCH_PARAMS['input_size'] = [4]
#   C.ARCH_PARAMS['output_size'] = [env.action_space.n]
#   agent = PGAgent(C)
#   agent.build_model(env)
#   render = False
#
#   running_length = 0
#   for i_episode in count(1):
#     state = env.reset()
#
#     episode_length=0
#     T=range(10000)
#     for t in T:
#       action, log_prob, entropy, *_ = agent.sample_action(state)
#       state, reward, done, _ = env.step(action)
#
#       if render:
#         env.render()
#
#       agent.trajectory.remember(reward, log_prob, entropy)
#       if done:
#         episode_length = t
#         break
#
#     running_length = running_length * 0.99 + episode_length * 0.01
#     error = agent.evaluate_model()
#     agent.trajectory.forget()
#     if error is not None:
#       if agent.use_batched_updates:
#         agent.accum_error += error
#         if agent._rollout_i % agent.batch_size == 0:
#           agent.optimise_wrt(agent.accum_error / agent.batch_size)
#           agent.accum_error = U.to_var([0.0], use_cuda=agent.C.USE_CUDA_IF_AVAILABLE)
#       else:
#         agent.optimise_wrt(error)
#
#     if i_episode % 10 == 0:
#       print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
#           i_episode, episode_length, running_length))
#
#     if running_length > env.spec.reward_threshold:
#       print("Solved! Running reward is now {} and "
#             "the last episode runs to {} time steps!".format(running_length,
#                                                              episode_length))
#       torch.save(agent._model.state_dict(), 'policy.model')
#       break


if __name__ == '__main__':
  import argparse
  import configs.pg_config as C

  parser = argparse.ArgumentParser(description='PG Agent')
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
  parser.add_argument('--SEED', '-S', type=int, default=1, metavar='SEED',
                      help='random seed (default: 1)')
  args = parser.parse_args()

  for k, arg in args.__dict__.items():
    setattr(C, k, arg)

  for k, arg in U.get_upper_vars_of(C).items():
    print(f'{k} = {arg}')

  input('\nPress any key to begin... ')

  try:
    test_pg_agent(C)
  except KeyboardInterrupt:
    print('Stopping')
