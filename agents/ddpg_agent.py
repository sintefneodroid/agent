#!/usr/bin/env python3
# coding=utf-8
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

  def __init__(self, config):
    super().__init__()
    self._step_n = 0
    self._rollout_i = 0

    self._use_cuda = config.USE_CUDA_IF_AVAILABLE
    self._optimiser_type = config.OPTIMISER_TYPE

    self._actor_optimiser_spec = U.OSpec(
        constructor=self._optimiser_type,
        kwargs=dict(lr=config.ACTOR_LEARNING_RATE),
        )
    self._critic_optimiser_spec = U.OSpec(
        constructor=self._optimiser_type,
        kwargs=dict(lr=config.CRITIC_LEARNING_RATE, weight_decay=config.Q_WEIGHT_DECAY),
        )

    self._random_process = OrnsteinUhlenbeckProcess(theta=config.THETA, sigma=config.SIGMA)

    # self._memory = U.PrioritisedReplayMemory(config.REPLAY_MEMORY_SIZE)  # Cuda trouble
    self._memory = U.ReplayMemory(config.REPLAY_MEMORY_SIZE)
    self._evaluation_function = config.EVALUATION_FUNCTION

    self._actor_arch = config.ACTOR_ARCH
    self._actor_arch_parameters = config.ACTOR_ARCH_PARAMS
    self._critic_arch = config.CRITIC_ARCH
    self._critic_arch_parameters = config.CRITIC_ARCH_PARAMS

    self._C = config

    self._input_size = config.ACTOR_ARCH_PARAMS['input_size']
    self._output_size = config.ACTOR_ARCH_PARAMS['output_size']
    self._batch_size = config.BATCH_SIZE
    self._value_type = config.VALUE_TENSOR_TYPE
    self._discount_factor = config.DISCOUNT_FACTOR
    self._learning_rate = config.LEARNING_RATE
    self._epsilon = config.EPSILON
    self._use_double_dqn = config.DOUBLE_DQN
    self._signal_clipping = config.SIGNAL_CLIPPING
    self._initial_observation_period = config.INITIAL_OBSERVATION_PERIOD
    self._learning_frequency = config.LEARNING_FREQUENCY
    self._sync_target_model_frequency = config.SYNC_TARGET_MODEL_FREQUENCY
    self._state_tensor_type = config.STATE_TENSOR_TYPE

    self._eps_start = config.EPS_START
    self._eps_end = config.EPS_END
    self._eps_decay = config.EPS_DECAY

    self._early_stopping_condition = None
    self._model = None
    self._target_model = None
    self._optimiser = None

    self._end_training = False

    self._batch_size = config.BATCH_SIZE
    self._tau = config.TAU
    self._gamma = config.GAMMA

    self._end_training = False

    # Construct the replay memory
    self._replay_memory = U.ReplayMemory(config.REPLAY_MEMORY_SIZE)

  def stop_training(self):
    self._end_training = True

  def save_model(self, C):
    U.save_model(self.actor, C)
    # U.save_model(self.critic, C)

  def __build_models__(self):

    # Construct actor and critic
    actor = self._actor_arch(self._actor_arch_parameters).type(torch.FloatTensor)
    target_actor = self._actor_arch(self._actor_arch_parameters).type(torch.FloatTensor)
    critic = self._critic_arch(self._critic_arch_parameters).type(torch.FloatTensor)
    target_critic = self._critic_arch(self._critic_arch_parameters).type(torch.FloatTensor)

    if self._use_cuda:
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
    if type(self._input_size) is str:
      self._input_size = env.observation_space.shape
    print('observation dimensions: ', self._input_size)

    if type(self._output_size) is str:
      if len(env.action_space.shape) >= 1:
        self._output_size = env.action_space.shape
      else:
        self._output_size = [env.action_space.n]
    print('action dimensions: ', self._output_size)

    self._actor_arch_parameters['input_size'] = self._input_size
    self._actor_arch_parameters['output_size'] = self._output_size
    self._critic_arch_parameters['input_size'] = self._input_size
    self._critic_arch_parameters['output_size'] = self._output_size

    self.actor, self.target_actor, self.critic, self.target_critic, self.actor_optimizer, \
    self.critic_optimizer = self.__build_models__()

  def sample_model(self, state):
    state = U.to_var(state, volatile=True, use_cuda=self._use_cuda).unsqueeze(0)
    action = self.actor(state)
    return action.data.cpu()[0].numpy()

  def sample_action(self, state):
    return self.sample_model(state)

  def update(self, gamma=1.0):

    if len(self._replay_memory) < self._batch_size:
      return
    state_batch, action_batch, signal_batch, next_state_batch, non_terminal_batch = \
      self._replay_memory.sample_transitions(self._batch_size)

    state_batch = U.to_var(state_batch, use_cuda=self._use_cuda).view(-1, self._input_size[0])
    next_state_batch = U.to_var(next_state_batch, use_cuda=self._use_cuda).view(-1, self._input_size[0])
    action_batch = U.to_var(action_batch, use_cuda=self._use_cuda).view(-1, self._output_size[0])
    signal_batch = U.to_var(signal_batch, use_cuda=self._use_cuda).unsqueeze(0)
    non_terminal_mask = U.to_var(non_terminal_batch, use_cuda=self._use_cuda).unsqueeze(0)

    ### Critic ###
    # Compute current Q value, critic takes state and action choosen
    Q_current = self.critic(state_batch, action_batch)
    # Compute next Q value based on which action target actor would choose
    # Detach variable from the current graph since we don't want gradients for next Q to propagated
    target_actions = self.target_actor(state_batch)
    next_max_q = self.target_critic(next_state_batch, target_actions).detach().max(1)[0]

    next_Q_values = non_terminal_mask * next_max_q
    Q_target = signal_batch + (gamma * next_Q_values)  # Compute the target of the current Q values
    critic_loss = F.smooth_l1_loss(Q_current, Q_target)  # Compute Bellman error (using Huber loss)

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()  # Optimize the critic

    ### Actor ###
    actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()  # Optimize the actor

    self.update_target(self.target_critic, self.critic)
    self.update_target(self.target_actor, self.actor)  # Update the target networks

  def update_target(self, target_model, model):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
      target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

  def rollout(self, initial_state, environment, render=False):
    self._rollout_i += 1

    state = initial_state
    episode_signal = 0
    episode_length = 0
    episode_q_error = 0

    T = tqdm(count(1), f'Rollout #{self._rollout_i}', leave=False)
    for t in T:
      self._step_n += 1

      action = self.sample_action(state)

      noise = self._random_process.sample()

      action += noise  # Add action space noise for exploration, alternative is parameter space noise

      if self._C.ACTION_CLIPPING:
        action = np.clip(action, -1.0, 1.0)

      next_state, signal, terminated, info = environment.step(action)

      if render:
        environment.render()

      if self._C.SIGNAL_CLIPPING:
        signal = np.clip(action, -1.0, 1.0)

      successor_state = None
      if not terminated:  # If environment terminated then there is no successor state
        successor_state = next_state

      self._replay_memory.add_transition(state,
                                         action,
                                         signal,
                                         successor_state,
                                         not terminated)
      state = next_state

      self.update(self._gamma)
      episode_signal += signal

      if terminated:
        episode_length = t
        break

    return episode_signal, episode_length

  def train(self, env, rollouts=1000, render=False, render_frequency=10, stat_frequency=10):
    '''The Deep Deterministic Policy Gradient algorithm.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    random_process: Defined in utils.random_process
        The process that add noise for exploration in deterministic policy.
    agent:
        a DDPG agent consists of a actor and critic.
    num_episodes:
        Number of episodes to run for.
    gamma: float
        Discount Factor
    log_every_n_eps: int
        Log and plot training info every n episodes.
        :param rollouts:
        :type rollouts:
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
        term_plot([i for i in range(1, episode_i + 1)], stats.signal_mas, E.write, offset=0)
        E.set_description(f'Episode: {episode_i}, Moving signal: {stats.signal_mas[-1]}')

      signal, dur = 0, 0
      if render and episode_i % render_frequency == 0:
        signal, dur, *rolloutstats = self.rollout(state, env, render=render)
      else:
        signal, dur, *rolloutstats = self.rollout(state, env)

      stats.episode_lengths.append(dur)
      stats.episode_signals.append(signal)
      signal_ma = np.mean(stats.episode_signals[-100:])
      stats.signal_mas.append(signal_ma)

      if self._end_training:
        break

    return self.actor, self.critic, stats

  # def train2(self, env, rollouts=1000, render=False, render_frequency=100, stat_frequency=100):
  #   '''The Deep Deterministic Policy Gradient algorithm.
  #   Parameters
  #   ----------
  #   env: gym.Env
  #       gym environment to train on.
  #   random_process: Defined in utils.random_process
  #       The process that add noise for exploration in deterministic policy.
  #   agent:
  #       a DDPG agent consists of a actor and critic.
  #   num_episodes:
  #       Number of episodes to run for.
  #   gamma: float
  #       Discount Factor
  #   log_every_n_eps: int
  #       Log and plot training info every n episodes.
  #   '''
  #   stats = U.plotting.EpisodeStatistics(
  #       episode_lengths=[],
  #       episode_signals=[],
  #       signal_mas=[])
  #
  #   steps = 0
  #
  #   for episode_i in range(rollouts):
  #     state = env.reset()
  #
  #     self.random_process.reset()
  #
  #     episode_signal = 0
  #
  #     E = count(1)
  #     E = tqdm(E, f'Episode: {episode_i}', leave=True)
  #
  #     for t in E:
  #       action = self.sample_action(state)
  #
  #       noise = self.random_process.sample()
  #
  #       action += noise  # Add action space noise for exploration, alternative is parameter space noise
  #
  #       if self.C.ACTION_CLIPPING:
  #         action = np.clip(action, -1.0, 1.0)
  #
  #       next_state, signal, terminated, _ = env.step(action)
  #       if render:
  #         env.render()
  #
  #       steps += 1
  #       episode_signal += signal
  #       episode_length = t
  #
  #       self.replay_memory.add_transition(state,
  #                                         action,
  #                                         signal,
  #                                         next_state,
  #                                         not terminated)
  #       state = next_state
  #
  #       self.update(self.gamma)
  #
  #       if terminated:
  #         stats.episode_lengths.append(episode_length)
  #         stats.episode_signals.append(episode_signal)
  #         signal_ma = np.mean(stats.episode_signals[-100:])
  #         stats.signal_mas.append(signal_ma)
  #         if episode_i % 10 == 0:
  #           print(f'*** EPISODE {episode_i + 1} *** {stats.episode_lengths[episode_i]} TIMESTEPS')
  #           print('MEAN REWARD (100 episodes): ' + '%.3f' % (signal_ma))
  #           print('TOTAL TIMESTEPS SO FAR: %d' % (steps))
  #           # U.plotting.plot_episode_stats(stats)
  #         break
  #
  #   return self.actor, self.critic, stats

  def infer(self, environment, render=True):

    for episode_i in count(1):
      print('Episode {}'.format(episode_i))
      state = environment.reset()

      for t in count(1):

        a = self.sample_model(state)
        state, reward, terminated, info = environment.step(a)
        if render:
          environment.render()

        if terminated:
          break

  def load_model(self, C, model_path):
    print('loading latest model: ' + model_path)
    self.actor = U.ActorArchitecture(C).type(torch.FloatTensor)
    self.actor.load_state_dict(torch.load(model_path))
    self.actor = self.actor.eval()  # _model.train(False)
    if C.USE_CUDA_IF_AVAILABLE:
      self.actor = self.actor.cuda()
    else:
      self.actor = self.actor.cpu()


def test_ddpg_agent():
  import configs.ddpg_config as C
  from utilities.environment_wrappers.normalise_actions import NormaliseActionsWrapper
  import gym

  env = NormaliseActionsWrapper(gym.make('Pendulum-v0'))
  # env = neo.make('satellite',connect_to_running=False)

  agent = DDPGAgent(C)
  listener = U.add_early_stopping_key_combination(agent.stop_training)

  listener.start()
  try:
    agent.build_model(env)
    actor_model, critic_model, stats = agent.train(env, C.EPISODES, render=True)
  finally:
    listener.stop()

  U.save_model(actor_model, C, name='actor')
  U.save_model(critic_model, C, name='critic')


if __name__ == '__main__':
  test_ddpg_agent()
