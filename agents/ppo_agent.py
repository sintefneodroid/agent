#!/usr/local/bin/python
# coding: utf-8
__author__ = 'cnheider'

import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

import utilities as U
from agents.ac_agent import ACAgent

cv2.setNumThreads(0)


class PPOAgent(ACAgent):
  '''
An agent learned with PPO using Advantage Actor-Critic framework
- Actor takes state as input
- Critic takes both state and action as input
- agent interact with environment to collect experience
- agent training with experience to update policy
- adam seems better than rmsprop for ppo
'''

  def rollout(self, init_obs, env, **kwargs):
    pass

  def __local_defaults__(self):
    self._steps = 10

    self._discount_factor = 0.99
    self._gae_tau = 0.95
    self._reached_horizon_penalty = -10.

    self._experience_buffer = U.ExpandableBuffer()
    self._critic_loss = nn.MSELoss
    self._actor_critic_lr = 3e-4
    self._entropy_reg_coef = 0.1
    self._value_reg_coef = 1.
    self._batch_size = 10
    self._initial_observation_period = 0
    self._target_update_tau = 1.0
    self._update_target_interval = 1000
    self._max_grad_norm = None

    # params for epsilon greedy
    self._epsilon_start = 0.99
    self._epsilon_end = 0.05
    self._epsilon_decay = 500

    self._use_cuda = False

    self._surrogate_clip = 0.2

    self._optimiser_type = torch.optim.Adam

    self._actor_critic_arch = U.ActorCriticNetwork
    self._actor_critic_arch_params = {
      'input_size':              None,
      'hidden_size':             [32, 32],
      'actor_hidden_size':       [32],
      'critic_hidden_size':      [32],
      'actor_output_size':       None,
      'actor_output_activation': F.log_softmax,
      'critic_output_size':      [1],
      'continuous':              True,
      }

    self._actor_critic = None
    self._actor_critic_target = None
    self._optimiser = None

  def __build_models__(self):
    self._actor_critic_arch_params['input_size'] = self._input_size
    self._actor_critic_arch_params['actor_output_size'] = self._output_size

    actor_critic = self._actor_critic_arch(**self._actor_critic_arch_params)

    actor_critic_target = self._actor_critic_arch(**self._actor_critic_arch_params)
    actor_critic_target.load_state_dict(actor_critic.state_dict())

    if self._use_cuda:
      actor_critic.cuda()
      actor_critic_target.cuda()

    optimiser = self._optimiser_type(
        actor_critic.parameters(), lr=self._actor_critic_lr
        )

    self._actor_critic, self._actor_critic_target, self._optimiser = actor_critic, actor_critic_target, \
                                                                     optimiser

  def maybe_take_n_steps(self, initial_state, environment, n=100, render=False):
    state = initial_state
    accumulated_signal = 0

    transitions = []
    terminated = False

    T = tqdm(range(1, n + 1), f'Step #{self._step_i}', leave=False)
    for t in T:
      self._step_i += 1
      action, value_estimates, action_prob, *_ = self.sample_action(state)

      next_state, signal, terminated, _ = environment.step(action.data[0])

      if render:
        environment.render()

      successor_state = None
      if not terminated:  # If environment terminated then there is no successor state
        successor_state = next_state

      transitions.append(
          U.ValuedTransition(
              state,
              action,
              action_prob,
              value_estimates,
              signal,
              successor_state,
              not terminated,
              )
          )

      state = next_state

      accumulated_signal += signal

      if self._step_i % self._update_target_interval == 0:
        self._actor_critic_target.load_state_dict(
            self._actor_critic.state_dict()
            )

      if terminated:
        break

    return transitions, accumulated_signal, terminated, state

  def trace_back_steps(self, transitions):
    n_step_summary = U.ValuedTransition(*zip(*transitions))

    advantages, discounted_returns = U.generalised_advantage_estimate(
        n_step_summary, self._discount_factor, gae_tau=self._gae_tau
        )

    i = 0
    advantage_memories = []
    for step in zip(*n_step_summary):
      step = U.ValuedTransition(*step)
      advantage_memories.append(
          U.AdvantageMemory(
              step.state,
              step.action,
              step.action_prob,
              step.value_estimate,
              advantages[i],
              discounted_returns[i],
              )
          )
      i += 1

    return advantage_memories

  def evaluate(self, batch, **kwargs):

    states_var = U.to_var(batch.state, use_cuda=self._use_cuda).view(
        -1, self._input_size[0]
        )
    action_var = U.to_var(batch.action, use_cuda=self._use_cuda, dtype='long').view(
        -1, self._output_size[0]
        )
    action_probs_var = U.to_var(batch.action_prob, use_cuda=self._use_cuda).view(
        -1, self._output_size[0]
        )
    values_var = U.to_var(batch.value_estimate, use_cuda=self._use_cuda)
    advantages_var = U.to_var(batch.advantage, use_cuda=self._use_cuda)
    returns_var = U.to_var(batch.discounted_return, use_cuda=self._use_cuda)

    value_error = (.5 * (values_var - returns_var) ** 2.).mean()

    entropy_loss = U.entropy(action_probs_var).mean()

    action_prob = action_probs_var  # .gather(1, action_var)
    _, _, action_probs_target, *_ = self._actor_critic_target(states_var)
    action_prob_target = action_probs_target  # .gather(1, action_var)
    # ratio = action_prob / (action_prob_target + self._divide_by_zero_safety)
    ratio = torch.exp(action_prob - action_prob_target)

    advantage = (advantages_var - advantages_var.mean()) / (
        advantages_var.std() + self._divide_by_zero_safety
    )

    surrogate = ratio * advantage
    surrogate_clipped = torch.clamp(
        ratio, min=1. - self._surrogate_clip, max=1. + self._surrogate_clip
        ) * advantage  # (L^CLIP)

    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

    cost = policy_loss + value_error * self._value_reg_coef + entropy_loss * self._entropy_reg_coef

    return cost

  def train(self):
    batch = U.AdvantageMemory(*zip(*self._experience_buffer.memory))
    cost = self.evaluate(batch)
    self.__optimise_wrt__(cost)

  def __optimise_wrt__(self, cost, **kwargs):
    self._optimiser.zero_grad()
    cost.backward()
    if self._max_grad_norm is not None:
      nn.utils.clip_grad_norm(
          self._actor_critic.parameters(), self._max_grad_norm
          )
    self._optimiser.step()

  def __sample_model__(self, state, continuous=True, **kwargs):
    '''

continuous
  randomly sample from normal distribution, whose mean and variance come from policy network.
  [batch, action_size]

:param state:
:type state:
:param continuous:
:type continuous:
:param kwargs:
:type kwargs:
:return:
:rtype:
'''
    state_var = U.to_var(state, use_cuda=self._use_cuda, unsqueeze=True)

    if continuous:

      action_mean, action_log_std, value_estimate = self._actor_critic(state_var)

      action = torch.normal(action_mean, torch.exp(action_log_std))

      # value, x, states = self(inputs, states, masks)
      # action = self.dist.sample(x, deterministic=deterministic)
      # action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)

      return action, value_estimate, action_log_std
    else:

      softmax_probs, value_estimate = self._actor_critic(state_var)
      m = Categorical(softmax_probs)
      action = m.sample()
      a = action.cpu().data.numpy()[0]
      return a, value_estimate, m.log_prob(action), m.probs

  # choose an action based on state for execution
  def sample_action(self, state, **kwargs):
    action, value_estimate, action_log_std, *_ = self.__sample_model__(state)
    return action, value_estimate, action_log_std


def test_ppo_agent(config):
  import gym

  device = torch.device('cuda' if config.USE_CUDA else 'cpu')
  U.set_seed(config.SEED)

  env = gym.make(config.ENVIRONMENT_NAME)
  env.seed(config.SEED)

  ppo_agent = PPOAgent(config)
  ppo_agent.build_agent(env, device)

  initial_state = env.reset()
  cs = tqdm(range(1, config.ROLLOUTS + 1), f'Rollout {0}, {0}', leave=False)
  for rollout_i in cs:

    transitions, accumulated_signal, terminated, initial_state = ppo_agent.maybe_take_n_steps(
        initial_state, env, render=config.RENDER_ENVIRONMENT
        )

    if terminated:
      initial_state = env.reset()

    if rollout_i >= config.INITIAL_OBSERVATION_PERIOD:
      advantage_memories = ppo_agent.trace_back_steps(transitions)
      for m in advantage_memories:
        ppo_agent._experience_buffer.remember(m)

      ppo_agent.train()
      ppo_agent._experience_buffer.forget()


if __name__ == '__main__':
  import configs.ppo_config as C

  from configs.arguments import parse_arguments

  args = parse_arguments('PPO Agent',C)

  for k, arg in args.__dict__.items():
    setattr(C, k, arg)

  print(f'Using config: {C}')
  if not args.skip_confirmation:
    for k, arg in U.get_upper_vars_of(C).items():
      print(f'{k} = {arg}')
    input('\nPress any key to begin... ')

  try:
    test_ppo_agent(C)
  except KeyboardInterrupt:
    print('Stopping')

  torch.cuda.empty_cache()
