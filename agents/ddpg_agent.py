#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'
from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import utilities as U
from agents.ac_agent import ACAgent
from utilities.random_process.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess
from utilities.visualisation.term_plot import term_plot


class DDPGAgent(ACAgent):
  '''
  The Deep Deterministic Policy Gradient (DDPG) Agent
  Parameters
  ----------
      actor_optimizer_spec: OptimizerSpec
          Specifying the constructor and kwargs, as well as learning rate and other
          parameters for the optimizer
      critic_optimizer_spec: OptimizerSpec
      num_feature: int
          The number of features of the environmental state
      num_action: int
          The number of available actions that agent can choose from
      replay_memory_size: int
          How many memories to store in the replay memory.
      batch_size: int
          How many transitions to sample each time experience is replayed.
      tau: float
          The update rate that target networks slowly track the learned networks.
  '''

  def optimise_wrt(self, td_error, state_batch, **kwargs):
    """

    :type kwargs: object
    """
    self.optimise_critic_wrt(td_error)

    ### Actor ###
    loss = -self._critic(state_batch, self._actor(state_batch)).mean()
    # loss = -torch.sum(self.critic(state_batch, self.actor(state_batch)))

    self.optimise_actor_wrt(loss)

    self.update_target(self._target_critic, self._critic)
    self.update_target(self._target_actor, self._actor)

    # self._memory.batch_update(indices, errors.tolist())  # Cuda trouble

    return loss

  def __defaults__(self):
    self._optimiser_type = torch.optim.Adam

    self._actor_optimiser_spec = U.OSpec(
        constructor=self._optimiser_type,
        kwargs=dict(lr=0.0001),
        )
    self._critic_optimiser_spec = U.OSpec(
        constructor=self._optimiser_type,
        kwargs=dict(lr=0.001, weight_decay=0.01),
        )

    self._random_process = OrnsteinUhlenbeckProcess(theta=0.15, sigma=0.2)
    # Adds noise for exploration

    # self._memory = U.PrioritisedReplayMemory(config.REPLAY_MEMORY_SIZE)  # Cuda trouble
    self._memory = U.ReplayMemory(1000000)
    self._evaluation_function = F.smooth_l1_loss

    self._actor_arch = U.ActorArchitecture
    self._actor_arch_parameters = {
      'input_size':        None,  # Obtain from environment
      'hidden_size':       [128, 64],
      'output_activation': None,
      'output_size':       None  # Obtain from environment
      }

    self._critic_arch = U.CriticArchitecture
    self._critic_arch_parameters = {
      'input_size':        None,  # Obtain from environment
      'hidden_size':       [128, 64],
      'output_activation': None,
      'output_size':       None  # Obtain from environment
      }

    self._discount_factor = 0.99
    self._use_double_dqn = False
    self._signal_clipping = False
    self._action_clipping = False
    self._initial_observation_period = 10000
    self._learning_frequency = 4
    self._sync_target_model_frequency = 10000
    self._state_tensor_type = torch.FloatTensor
    self._value_tensor_type = torch.FloatTensor

    self._epsilon_start = 0.9
    self._epsilon_end = 0.05
    self._epsilon_decay = 35000

    self._early_stopping_condition = None
    self._optimiser = None

    self._end_training = False

    self._batch_size = 60
    self._target_update_tau = 0.001

    self._end_training = False

    self._actor, self._target_actor, self._critic, self._target_critic, self._actor_optimiser, \
    self._critic_optimiser = None, None, None, None, None, None

    self._input_size = None
    self._output_size = None

  def save_model(self, C):
    U.save_model(self._actor, C, 'actor')
    U.save_model(self._critic, C, 'policy')

  def load_model(self, model_path, evaluation=False):
    print('loading latest model: ' + model_path)

    self._actor, self._target_actor, self._critic, self._target_critic, self._actor_optimiser, \
    self._critic_optimiser = self.__build_models__()

    self._actor.load_state_dict(torch.load(f'actor-{model_path}'))
    self._critic.load_state_dict(torch.load(f'critic-{model_path}'))

    self.update_target(self._target_critic, self._critic)
    self.update_target(self._target_actor, self._actor)

    if evaluation:
      self._actor = self._actor.eval()
      self._actor.train(False)

    if self._use_cuda_if_available:
      self._actor = self._actor.cuda()
      self._target_actor = self._target_actor.cuda()
      self._critic = self._critic.cuda()
      self._target_critic = self._target_critic.cuda()
    else:
      self._actor = self._actor.cpu()
      self._target_actor = self._target_actor.cpu()
      self._critic = self._critic.cpu()
      self._target_critic = self._target_critic.cpu()

  def __build_models__(self):

    # Construct actor and critic
    actor = self._actor_arch(**self._actor_arch_parameters).type(torch.FloatTensor)
    target_actor = self._actor_arch(**self._actor_arch_parameters).type(torch.FloatTensor)
    critic = self._critic_arch(**self._critic_arch_parameters).type(torch.FloatTensor)
    target_critic = self._critic_arch(**self._critic_arch_parameters).type(torch.FloatTensor)

    if self._use_cuda_if_available:
      actor = actor.cuda()
      target_actor = target_actor.cuda()
      critic = critic.cuda()
      target_critic = target_critic.cuda()

    # Construct the optimizers for actor and critic
    actor_optimizer = self._actor_optimiser_spec.constructor(actor.parameters(),
                                                             **self._critic_optimiser_spec.kwargs)
    critic_optimizer = self._critic_optimiser_spec.constructor(critic.parameters(),
                                                               **self._critic_optimiser_spec.kwargs)

    return actor, target_actor, critic, target_critic, actor_optimizer, critic_optimizer

  def build_model(self, env):
    self.infer_input_output_sizes(env)

    self._actor_arch_parameters['input_size'] = self._input_size
    self._actor_arch_parameters['output_size'] = self._output_size
    self._critic_arch_parameters['input_size'] = self._input_size
    self._critic_arch_parameters['output_size'] = self._output_size

    self._actor, self._target_actor, self._critic, self._target_critic, self._actor_optimiser, \
    self._critic_optimiser = self.__build_models__()

  def sample_model(self, state, **kwargs):
    state = U.to_var(state, volatile=True, use_cuda=self._use_cuda_if_available).unsqueeze(0)
    action = self._actor(state)
    return action.data.cpu()[0].numpy()

  def sample_action(self, state, **kwargs):
    return self.sample_model(state)

  def evaluate(self,
               state_batch,
               action_batch,
               signal_batch,
               next_state_batch,
               non_terminal_batch,
               **kwargs):

    state_batch_var = U.to_var(state_batch, use_cuda=self._use_cuda_if_available).view(-1, self._input_size[0])
    next_state_batch_var = U.to_var(next_state_batch, use_cuda=self._use_cuda_if_available).view(-1,
                                                                                                 self._input_size[
                                                                                                 0])
    action_batch_var = U.to_var(action_batch, use_cuda=self._use_cuda_if_available).view(-1,
                                                                                         self._output_size[0])
    signal_batch_var = U.to_var(signal_batch, use_cuda=self._use_cuda_if_available).unsqueeze(0)
    non_terminal_mask_var = U.to_var(non_terminal_batch, use_cuda=self._use_cuda_if_available).unsqueeze(0)

    ### Critic ###
    # Compute current Q value, critic takes state and action choosen
    Q_current = self._critic(state_batch_var, action_batch_var)
    # Compute next Q value based on which action target actor would choose
    # Detach variable from the current graph since we don't want gradients for next Q to propagated
    target_actions = self._target_actor(state_batch_var)
    next_max_q = self._target_critic(next_state_batch_var, target_actions).detach().max(1)[0]

    next_Q_values = non_terminal_mask_var * next_max_q
    Q_target = signal_batch_var + (
          self._discount_factor * next_Q_values)  # Compute the target of the current Q values
    td_error = F.smooth_l1_loss(Q_current, Q_target)  # Compute Bellman error (using Huber loss)

    return td_error, state_batch_var

  def update_models(self):
    """
    Update the target networks

    :return:
    :rtype:
    """
    if len(self._memory) < self._batch_size:
      return

    batch = self._memory.sample_transitions(self._batch_size)

    td_error, state_batch_var = self.evaluate(*batch)

    loss = self.optimise_wrt(td_error, state_batch_var)

    return td_error.data[0], loss.data[0]

  def optimise_critic_wrt(self, error):
    self._critic_optimiser.zero_grad()
    error.backward()
    self._critic_optimiser.step()  # Optimize the critic

  def optimise_actor_wrt(self, loss):
    self._actor_optimiser.zero_grad()
    loss.backward()
    self._actor_optimiser.step()  # Optimize the actor

  def update_target(self, target_model, model):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
      target_param.data.copy_(self._target_update_tau * param.data + (1 - self._target_update_tau) * target_param.data)

  def rollout(self, initial_state, environment, render=False):
    self._rollout_i += 1

    state = initial_state
    episode_signal = 0
    episode_length = 0

    T = tqdm(count(1), f'Rollout #{self._rollout_i}', leave=False)
    for t in T:
      self._step_i += 1

      action = self.sample_action(state)

      noise = self._random_process.sample()

      action += noise  # Add action space noise for exploration, alternative is parameter space noise

      if self._action_clipping:
        action = np.clip(action, -1.0, 1.0)

      next_state, signal, terminated, info = environment.step(action)

      if render:
        environment.render()

      if self._action_clipping:
        signal = np.clip(action, -1.0, 1.0)

      # successor_state = None
      # if not terminated:  # If environment terminated then there is no successor state
      successor_state = next_state

      self._memory.add_transition(state,
                                  action,
                                  signal,
                                  successor_state,
                                  not terminated)
      state = next_state

      self.update_models()
      episode_signal += signal

      if terminated:
        episode_length = t
        break

    return episode_signal, episode_length

  def train(self, env, rollouts=1000, render=False, render_frequency=10, stat_frequency=10):
    '''
    The Deep Deterministic Policy Gradient algorithm.

    :param env: gym environment to train on.
    :type env: gym.Env
    :param rollouts: Number of episodes to run for.
    :type rollouts: int
    :param render:
    :type render:
    :param render_frequency:
    :type render_frequency:
    :param stat_frequency:
    :type stat_frequency:
    '''
    stats = U.plotting.EpisodeStatistics(
        episode_lengths=[],
        episode_signals=[],
        signal_mas=[])

    E = range(1, rollouts)
    E = tqdm(E, desc='', leave=True)

    for episode_i in E:
      state = env.reset()
      self._random_process.reset()

      if episode_i % stat_frequency == 0:
        t_episode = [i for i in range(1, episode_i + 1)]
        term_plot(t_episode, stats.signal_mas, 'Moving signal', printer=E.write)
        E.set_description(f'Episode: {episode_i}, Last Moving signal: {stats.signal_mas[-1]}')

      signal, dur = 0, 0
      if render and episode_i % render_frequency == 0:
        signal, dur, *rollout_stats = self.rollout(state, env, render=render)
      else:
        signal, dur, *rollout_stats = self.rollout(state, env)

      stats.episode_lengths.append(dur)
      stats.episode_signals.append(signal)
      signal_ma = np.mean(stats.episode_signals[-100:])
      stats.signal_mas.append(signal_ma)

      if self._end_training:
        break

    return self._actor, self._critic, stats


def test_ddpg_agent(config):
  """

  :rtype: object
  """
  from utilities.environment_wrappers.normalise_actions import NormaliseActionsWrapper
  import gym

  env = NormaliseActionsWrapper(gym.make(config.ENVIRONMENT_NAME))
  # env = neo.make('satellite',connect_to_running=False)

  agent = DDPGAgent(config)
  agent.build_model(env)
  listener = U.add_early_stopping_key_combination(agent.stop_training)

  listener.start()
  try:
    actor_model, critic_model, stats = agent.train(env, config.EPISODES, render=config.RENDER_ENVIRONMENT)
  finally:
    listener.stop()

  U.save_model(actor_model, config, name='actor')
  U.save_model(critic_model, config, name='critic')


if __name__ == '__main__':
  import configs.ddpg_config as C
  import argparse

  parser = argparse.ArgumentParser(description='DDPG Agent')
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
  parser.add_argument('--skip_confirmation', '-skip', action='store_true',
                      default=False,
                      help='Skip confirmation of config to be used')
  args = parser.parse_args()

  for k, arg in args.__dict__.items():
    setattr(C, k, arg)

  print(f'Using config: {C}')
  if not args.skip_confirmation:
    for k, arg in U.get_upper_vars_of(C).items():
      print(f'{k} = {arg}')
    input('\nPress any key to begin... ')

  try:
    test_ddpg_agent(C)
  except KeyboardInterrupt:
    print('Stopping')
