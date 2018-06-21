#!/usr/local/bin/python
# coding: utf-8
from itertools import count

__author__ = 'cnheider'

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

import utilities as U
from agents.abstract.joint_ac_agent import JointACAgent


class PPOAgent(JointACAgent):
  '''
'''

  def _defaults(self):
    self._steps = 10

    self._discount_factor = 0.99
    self._gae_tau = 0.95
    self._reached_horizon_penalty = -10.

    self._experience_buffer = U.ExpandableCircularBuffer()
    self._critic_loss = nn.MSELoss
    self._actor_critic_lr = 3e-4
    self._entropy_reg_coef = 0.1
    self._value_reg_coef = 1.
    self._batch_size = 2048
    self._initial_observation_period = 0
    self._target_update_tau = 1.0
    self._update_target_interval = 1000
    self._max_grad_norm = None

    self._state_type = torch.float
    self._value_type = torch.float
    self._action_type = torch.long

    # params for epsilon greedy
    self._epsilon_start = 0.99
    self._epsilon_end = 0.05
    self._epsilon_decay = 500

    self._use_cuda = False

    self._surrogate_clip = 0.2

    self._optimiser_type = torch.optim.Adam

    self._actor_critic_arch = U.ActorCriticNetwork
    self._actor_critic_arch_params = {
      'input_size':             None,
      'hidden_layers':          [32, 32],
      'actor_hidden_layers':    [32],
      'critic_hidden_layers':   [32],
      'actor_output_size':      None,
      'actor_output_activation':F.log_softmax,
      'critic_output_size':     [1],
      'continuous':             True,
      }

    self._actor_critic = None
    self._actor_critic_target = None
    self._optimiser = None

  def _build(self):
    self._actor_critic_arch_params['input_size'] = self._input_size
    self._actor_critic_arch_params['actor_output_size'] = self._output_size

    actor_critic = self._actor_critic_arch(**self._actor_critic_arch_params)

    actor_critic_target = self._actor_critic_arch(**self._actor_critic_arch_params)
    actor_critic_target = U.copy_state(actor_critic_target, actor_critic)

    actor_critic.to(self._device)
    actor_critic_target.to(self._device)

    optimiser = self._optimiser_type(actor_critic.parameters(), lr=self._actor_critic_lr)

    self._actor_critic, self._actor_critic_target, self._optimiser = (
      actor_critic, actor_critic_target, optimiser)

  def _optimise_wrt(self, cost, **kwargs):
    self._optimiser.zero_grad()
    cost.backward()
    if self._max_grad_norm is not None:
      nn.utils.clip_grad_norm(self._actor_critic.parameters(), self._max_grad_norm)
    self._optimiser.step()

  def _sample_model(self, state, continuous=True, **kwargs):
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

    model_input = U.to_tensor([state], device=self._device, dtype=self._state_type)

    if continuous:
      with torch.no_grad():
        action_mean, action_log_std, value_estimate = self._actor_critic(model_input)

        action_log_std = action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        action = torch.normal(action_mean, action_std)

        a = action.to('cpu').numpy()[0]
      return a, value_estimate, action_log_std
    else:

      softmax_probs, value_estimate = self._actor_critic(model_input)
      # action = torch.multinomial(softmax_probs)
      m = Categorical(softmax_probs)
      action = m.sample()
      a = action.to('cpu').data.numpy()[0]
      return a, value_estimate, m.log_prob(action)

  def take_n_steps(self,
                   initial_state,
                   environment,
                   n=100,
                   render=False,
                   render_frequency=100):
    state = initial_state

    accumulated_signal = 0

    transitions = []
    terminated = False
    T = tqdm(range(1, n + 1), f'Step #{self._step_i} - {0}/{n}', leave=False)
    for t in T:
      # T.set_description(f'Step #{self._step_i} - {t}/{n}')
      self._step_i += 1
      action, value_estimates, action_prob, *_ = self.sample_action(state)

      next_state, signal, terminated, _ = environment.step(action)

      if render and self._rollout_i % render_frequency == 0:
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

      if terminated:
        state = environment.reset()
        self._rollout_i += 1

    return transitions, accumulated_signal, terminated, state

  def rollout(self,
              initial_state,
              environment,
              render=False, train=True,
              **kwargs):
    self._rollout_i += 1

    state = initial_state
    episode_signal = 0
    terminated = False
    episode_length = 0
    transitions = []

    T = tqdm(count(1), f'Rollout #{self._rollout_i}', leave=False)
    for t in T:
      self._step_i += 1

      action, value_estimates, action_prob, *_ = self.sample_action(state)

      next_state, signal, terminated, _ = environment.step(action)

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

      episode_signal += signal

      if terminated:
        episode_length = t
        break

    return transitions, episode_signal, terminated, state, episode_length

  def trace_back_steps(self, transitions):
    n_step_summary = U.ValuedTransition(*zip(*transitions))

    advantages = U.generalised_advantage_estimate(n_step_summary, self._discount_factor, tau=self._gae_tau)

    value_estimates = U.to_tensor(n_step_summary.value_estimate, device=self._device, dtype=torch.float)

    discounted_returns = value_estimates + advantages

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

  def evaluate(self, batch, discrete=False, **kwargs):

    states = U.to_tensor(batch.state, device=self._device, dtype=torch.float).view(-1, self._input_size[0])

    value_estimates = U.to_tensor(batch.value_estimate, device=self._device, dtype=torch.float)

    advantages = U.to_tensor(batch.advantage, device=self._device, dtype=torch.float)

    discounted_returns = U.to_tensor(batch.discounted_return, device=self._device, dtype=torch.float)

    value_error = (value_estimates - discounted_returns).pow(2).mean()

    advantage = (advantages - advantages.mean()) / (advantages.std() + self._divide_by_zero_safety)

    action_probs = U.to_tensor(batch.action_prob, device=self._device, dtype=torch.float) \
      .view(-1, self._output_size[0])
    _, _, action_probs_target, *_ = self._actor_critic_target(states)

    if discrete:
      actions = U.to_tensor(batch.action, device=self._device, dtype=torch.float) \
        .view(-1, self._output_size[0])
      action_probs = action_probs.gather(1, actions)
      action_probs_target = action_probs_target.gather(1, actions)

    ratio = torch.exp(action_probs - action_probs_target)

    surrogate = ratio * advantage

    clamped_ratio = torch.clamp(ratio, min=1. - self._surrogate_clip, max=1. + self._surrogate_clip)
    surrogate_clipped = clamped_ratio * advantage  # (L^CLIP)

    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

    entropy_loss = U.entropy(action_probs).mean()

    collective_cost = policy_loss + value_error * self._value_reg_coef + entropy_loss * self._entropy_reg_coef

    return collective_cost, policy_loss, value_error

  def update(self):
    batch = U.AdvantageMemory(*zip(*self._experience_buffer.sample()))
    collective_cost, actor_loss, critic_loss = self.evaluate(batch)
    self._optimise_wrt(collective_cost)
    # self.__optimise_wrt_split__((actor_loss, critic_loss))

    '''
    def __optimise_wrt_split__(self, cost, **kwargs):
      (actor_loss, critic_loss) = cost

      self._critic_optimiser.zero_grad()
      critic_loss.backward()
      if self._max_grad_norm is not None:
        nn.utils.clip_grad_norm(
            self._critic.parameters(), self._max_grad_norm
            )
      self._critic_optimiser.step()

      self._actor_optimiser.zero_grad()
      actor_loss.backward()
      if self._max_grad_norm is not None:
        nn.utils.clip_grad_norm(
            self._actor.parameters(), self._max_grad_norm
            )
      self._actor_optimiser.step()
  '''

  # choose an action based on state for execution
  def sample_action(self, state, **kwargs):
    action, value_estimate, action_log_std, *_ = self._sample_model(state)
    return action, value_estimate, action_log_std

  def _train(self, *args, **kwargs):

    # num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    return self.train_episodic(*args, **kwargs)
    # return self.train_step_batched(*args, **kwargs)

  def train_step_batched(self,
                         env,
                         num_batches=10000,
                         render=False,
                         render_frequency=100,
                         stat_frequency=10,
                         batch_length=100):

    self._rollout_i = 1

    initial_state = env.reset()

    B = tqdm(range(1, num_batches + 1), f'Batch {0}, {num_batches} - Episode {self._rollout_i}', leave=False)
    for batch_i in B:
      if batch_i % stat_frequency == 0:
        B.set_description(f'Batch {batch_i}, {num_batches} - Episode {self._rollout_i}')

      if render and batch_i % render_frequency == 0:
        transitions, accumulated_signal, terminated, initial_state = self.take_n_steps(
            initial_state, env, render=render, n=batch_length
            )
      else:
        transitions, accumulated_signal, terminated, initial_state = self.take_n_steps(
            initial_state, env, n=batch_length
            )

      if batch_i >= self._initial_observation_period:
        advantage_memories = self.trace_back_steps(transitions)
        for m in advantage_memories:
          self._experience_buffer.add(m)

        self.update()
        self._experience_buffer.clear()

        if self._rollout_i % self._update_target_interval == 0:
          self._actor_critic_target.load_state_dict(
              self._actor_critic.state_dict()
              )

      if self._end_training:
        break

    return self._actor_critic, []

  def train_episodic(self,
                     env,
                     num_batches=10000,
                     render=False,
                     render_frequency=100,
                     stat_frequency=10,
                     **kwargs):

    self._rollout_i = 1

    initial_state = env.reset()

    B = tqdm(range(1, num_batches + 1), f'Batch {0}, {num_batches} - Episode {self._rollout_i}', leave=False)
    for batch_i in B:
      if batch_i % stat_frequency == 0:
        B.set_description(f'Batch {batch_i}, {num_batches} - Episode {self._rollout_i}')

      if render and batch_i % render_frequency == 0:
        transitions, accumulated_signal, terminated, *_ = self.rollout(
            initial_state, env, render=render
            )
      else:
        transitions, accumulated_signal, terminated, *_ = self.rollout(
            initial_state, env
            )

      initial_state = env.reset()

      if batch_i >= self._initial_observation_period:
        advantage_memories = self.trace_back_steps(transitions)

        for m in advantage_memories:
          self._experience_buffer.add(m)

        self.update()
        self._experience_buffer.clear()

        if self._rollout_i % self._update_target_interval == 0:
          self._actor_critic_target.load_state_dict(
              self._actor_critic.state_dict()
              )

      if self._end_training:
        break

    return self._actor_critic, []


if __name__ == '__main__':
  import configs.agent_test_configs.test_ppo_config as C

  U.test_agent_main(PPOAgent, C)
