#!/usr/local/bin/python
# coding: utf-8
from functools import wraps
from itertools import count
from typing import Any, Tuple

import gym
import numpy

import draugr

from procedures.agent_tests import test_agent_main

__author__ = 'cnheider'

import torch
from torch import nn, optim
from tqdm import tqdm

import utilities as U
from agents.abstract.ac_agent import ActorCriticAgent


class PPOAgent(ActorCriticAgent):
  '''
'''

  # region Private

  def __defaults__(self) -> None:
    self._steps = 10

    self._discount_factor = 0.99
    self._gae_tau = 0.95
    # self._reached_horizon_penalty = -10.

    self._critic_loss = nn.MSELoss
    self._actor_critic_lr = 3e-4
    self._entropy_reg_coef = 3e-4
    self._value_reg_coef = 0.5
    self._batch_size = 32
    self._mini_batch_size = 10
    self._initial_observation_period = 0
    self._target_update_tau = 1.0
    self._update_target_interval = 1000
    self._max_grad_norm = None
    self._solved_threshold = -200
    self._test_interval = 1000
    self._early_stop = False

    self._state_type = torch.float
    self._value_type = torch.float
    self._action_type = torch.long

    # params for epsilon greedy
    self._epsilon_start = 0.99
    self._epsilon_end = 0.05
    self._epsilon_decay = 500

    self._use_cuda = False

    self._surrogate_clipping_value = 0.2

    self._optimiser_type = torch.optim.Adam

    self._actor_arch = U.DDPGActorArchitecture
    self._actor_arch_parameters = {
      'input_size':       None,  # Obtain from environment
      'hidden_layers':    None,
      'output_activation':None,
      'output_size':      None,  # Obtain from environment
      }

    self._critic_arch = U.DDPGCriticArchitecture
    self._critic_arch_parameters = {
      'input_size':       None,  # Obtain from environment
      'hidden_layers':    None,
      'output_activation':None,
      'output_size':      None,  # Obtain from environment
      }

    self._optimiser = None

    self._update_early_stopping = None
    # self._update_early_stopping = self.kl_target_stop

  def kl_target_stop(self, old_log_probs, new_log_probs):

    '''
    TRPO

    negloss = -tf.reduce_mean(self.advantages_ph * tf.exp(self.logp - self.prev_logp))
    negloss += tf.reduce_mean(self.beta_ph * self.kl_divergence)
    negloss += tf.reduce_mean(self.ksi_ph * tf.square(tf.maximum(0.0, self.kl_divergence - 2 *
    self.kl_target)))

    self.ksi = 10

    Adaptive kl_target = 0.01
    Adaptive kl_target = 0.03

    :param old_log_probs:
    :param new_log_probs:
    :return:
    '''
    self.kl_target = 0.003
    self.beta = 1.0
    self.beta_max = 20
    self.beta_min = 1 / 20
    self.learning_rate = 1
    kl_now = torch.distributions.kl_divergence(old_log_probs, new_log_probs)
    if kl_now > 4 * self.kl_target:
      return True

    if kl_now < self.kl_target / 1.5:
      self.beta /= 2
    elif kl_now > self.kl_target * 1.5:
      self.beta *= 2
    self.beta = numpy.clip(self.beta, self.beta_min, self.beta_max)
    return False

  # endregion

  # region Protected

  def _optimise_wrt(self, cost, **kwargs):
    self._optimiser.zero_grad()

    cost.backward()

    if self._max_grad_norm is not None:
      nn.utils.clip_grad_norm(self._actor.parameters(), self._max_grad_norm)
      nn.utils.clip_grad_norm(self._critic.parameters(), self._max_grad_norm)

    self._optimiser.step()

  def _sample_model(self, state, **kwargs):
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

    model_input = U.to_tensor(state, device=self._device, dtype=self._state_type)

    mean, std = self._actor(model_input)
    value_estimate = self._critic(model_input)

    distribution = torch.distributions.Normal(mean, std)

    with torch.no_grad():
      action = distribution.sample()

    action_log_prob = distribution.log_prob(action)

    return action, action_log_prob, value_estimate, distribution

  def _train(self, *args, **kwargs):

    # num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    # return self.train_episodically(*args, **kwargs)
    return self.train_step_batched(*args, **kwargs)

  # endregion

  # region Public

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
      dist, value_estimates, *_ = self.sample_action(state)

      action = dist.sample()
      action_prob = dist.log_prob(action)

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

      dist, value_estimates, *_ = self.sample_action(state)

      action = dist.sample()
      action_prob = dist.log_prob(action)

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

  def back_trace(self, transitions):
    n_step_summary = U.ValuedTransition(*zip(*transitions))

    advantages = U.advantage_estimate(n_step_summary.signal,
                                      n_step_summary.non_terminal,
                                      n_step_summary.value_estimate,
                                      self._discount_factor,
                                      tau=self._gae_tau
                                      )

    value_estimates = U.to_tensor(n_step_summary.value_estimate, device=self._device)

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
    # region Tensorise

    states = U.to_tensor(batch.state, device=self._device).view(-1, self._input_size[0])

    value_estimates = U.to_tensor(batch.value_estimate, device=self._device)

    advantages = U.to_tensor(batch.advantage, device=self._device)

    discounted_returns = U.to_tensor(batch.discounted_return, device=self._device)

    action_probs = U.to_tensor(batch.action_prob, device=self._device) \
      .view(-1, self._output_size[0])

    # endregion

    advantage = (advantages - advantages.mean()) / (advantages.std() + self._divide_by_zero_safety)

    _, _, action_probs_old, *_ = self._sample_model(states)

    if discrete:
      actions = U.to_tensor(batch.action, device=self._device).view(-1, self._output_size[0])
      action_probs = action_probs.gather(1, actions)
      action_probs_old = action_probs_old.gather(1, actions)

    ratio = (action_probs - action_probs_old).exp()
    # Generated action probs from (new policy) and (old policy).
    # Values of [0..1] means that actions less likely with the new policy,
    # while values [>1] mean action a more likely now

    surrogate = ratio * advantage

    clamped_ratio = torch.clamp(ratio,
                                min=1. - self._surrogate_clipping_value,
                                max=1. + self._surrogate_clipping_value)
    surrogate_clipped = clamped_ratio * advantage  # (L^CLIP)

    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

    entropy_loss = U.entropy(action_probs).mean()

    value_error = (value_estimates - discounted_returns).pow(2).mean()

    collective_cost = policy_loss + value_error * self._value_reg_coef + entropy_loss * self._entropy_reg_coef

    return collective_cost, policy_loss, value_error

  def update(self):

    returns_ = U.compute_gae(self._last_value_estimate,
                             self._transitions_.signal,
                             self._transitions_.non_terminal,
                             self._transitions_.value_estimate)

    returns = torch.cat(returns_).view(-1, 1).detach()
    log_probs = torch.cat(self._transitions_.action_prob).detach()
    values = torch.cat(self._transitions_.value_estimate).detach()
    states = torch.cat(self._transitions_.state).view(-1, self._input_size[0])
    actions = torch.cat(self._transitions_.action)

    advantage = returns - values

    self.ppo_update(self._mini_batch_size,
                    states,
                    actions,
                    log_probs,
                    returns,
                    advantage)

    if self._rollout_i % self._update_target_interval == 0:
      self.update_target(target_model=self._target_actor,
                         source_model=self._actor,
                         target_update_tau=self._target_update_tau)
      self.update_target(target_model=self._target_critic,
                         source_model=self._critic,
                         target_update_tau=self._target_update_tau)

    '''
    batch = U.AdvantageMemory(*zip(*self._memory_buffer.sample()))
    collective_cost, actor_loss, critic_loss = self.evaluate(batch)

    self._optimise_wrt(collective_cost)
    '''

  def ppo_update(self,

                 mini_batch_size,
                 states, actions,
                 log_probs,
                 returns,
                 advantages,
                 ppo_epochs=6
                 ):
    mini_batch_gen = self.ppo_mini_batch_iter(mini_batch_size,
                                              states,
                                              actions,
                                              log_probs,
                                              returns,
                                              advantages)
    for _ in range(ppo_epochs):
      try:
        (state,
         action,
         old_log_probs,
         return_now,
         advantage) = mini_batch_gen.__next__()
      except StopIteration:
        return

      action_out, action_log_prob, value_estimate, distribution = self._sample_model(state)
      entropy = distribution.entropy().mean()
      new_log_probs = distribution.log_prob(action)

      ratio = (new_log_probs - old_log_probs).exp()
      surrogate = ratio * advantage
      surrogate_clipped = (torch.clamp(ratio,
                                       1.0 - self._surrogate_clipping_value,
                                       1.0 + self._surrogate_clipping_value)
                           * advantage)

      actor_loss = - torch.min(surrogate, surrogate_clipped).mean()
      critic_loss = (return_now - value_estimate).pow(2).mean()

      loss = self._value_reg_coef * critic_loss + actor_loss - self._entropy_reg_coef * entropy

      self._actor_optimiser.zero_grad()
      self._critic_optimiser.zero_grad()
      loss.backward()
      self._actor_optimiser.step()
      self._critic_optimiser.step()

      if self._update_early_stopping:
        if self._update_early_stopping(old_log_probs, new_log_probs):
          break

  #
  def sample_action(self, state, **kwargs) -> Tuple[Any, Any, Any]:
    action, action_log_prob, value_estimate,*_ = self._sample_model(state)

    return action, action_log_prob, value_estimate

  #
  def train_episodically(self,
                         env,
                         num_batches=10000,
                         render=False,
                         render_frequency=100,
                         stat_frequency=10,
                         **kwargs):

    self._rollout_i = 1

    initial_state = env.reset()

    B = tqdm(range(1, num_batches + 1), f'Batch {0}, {num_batches} - Rollout {self._rollout_i}', leave=False)
    for batch_i in B:
      if batch_i % stat_frequency == 0:
        B.set_description(f'Batch {batch_i}, {num_batches} - Rollout {self._rollout_i}')

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
        advantage_memories = self.back_trace(transitions)

        for memory in advantage_memories:
          self._memory_buffer.add(memory)

        self.update()
        self._memory_buffer.clear()

        if self._rollout_i % self._update_target_interval == 0:
          self.update_target(target_model=self._target_actor, source_model=self._actor,
                             target_update_tau=self._target_update_tau)
          self.update_target(target_model=self._target_critic, source_model=self._critic,
                             target_update_tau=self._target_update_tau)

      if self._end_training:
        break

    return self._actor, self._critic, []

  # endregion

  def train_step_batched(self,
                         environments,
                         *,
                         num_steps=40,
                         rollouts=1000000,
                         render=True,
                         num_test_env=1
                         ):
    self.stats = draugr.StatisticCollection(stats=('signal', 'test_signal', 'entropy'))

    state = environments.reset()

    S = tqdm(range(rollouts), leave=False).__iter__()
    while self._step_i < rollouts and not self._end_training:

      self.batch_signal = 0

      transitions = []

      successor_state = None

      state = U.to_tensor(state, device=self._device)

      I = tqdm(range(num_steps), leave=False).__iter__()
      for _ in I:

        action, action_log_prob, value_estimate = self.sample_action(state)

        a = action.to('cpu').numpy()[0]
        successor_state, signal, terminated, _ = environments.step(a)
        self.batch_signal += signal

        successor_state = U.to_tensor(successor_state, device=self._device)
        signal_ = U.to_tensor(signal, device=self._device)
        not_terminated = U.to_tensor(not terminated, device=self._device)

        transitions.append(
            U.ValuedTransition(state,
                               action,
                               action_log_prob,
                               value_estimate,
                               signal_,
                               successor_state,
                               not_terminated
                               )
            )

        state = successor_state

        self._step_i += 1
        S.__next__()

        if self._step_i % self._test_interval == 0:
          test_signals = [self.test_agent(_test_env, render=render) for _ in range(num_test_env)]
          test_signal = numpy.mean(test_signals)
          self.stats.test_signal.append(test_signal)

          draugr.terminal_plot(self.stats.signal.values, title='batch_signal', percent_size=(.8, .25))
          draugr.terminal_plot(self.stats.test_signal.values, title='test_signal', percent_size=(.8, .3))

          if test_signal > self._solved_threshold and self._early_stop:
            self._end_training = True

      self.stats.signal.append(self.batch_signal)

      # only calculate value of next state for the last step this time
      *_, self._last_value_estimate ,_ = self._sample_model(successor_state)

      self._transitions_ = U.ValuedTransition(*zip(*transitions))

      self.update()

    return self._actor, self._critic, []

  def old_batched_training(self):
    '''
        B = tqdm(range(1, num_updates + 1), f'Batch {0}, {num_batches} - Episode {self._rollout_i}',
        leave=False)
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
            advantage_memories = self.back_trace(transitions)
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

      '''
    pass

  def test_agent(self, test_environment, render=False):
    state = test_environment.reset()
    if render:
      test_environment.render()
    terminal = False
    total_signal = 0
    while not terminal:
      state = U.to_tensor(state, device=self._device).unsqueeze(0)
      with torch.no_grad():
        action, *_ = self.sample_action(state)
      next_state, signal, terminal, *_ = test_environment.step(action)
      state = next_state
      if render:
        test_environment.render()
      total_signal += signal.item()
    return total_signal

  @staticmethod
  def ppo_mini_batch_iter(mini_batch_size: int,
                          states: Any,
                          actions: Any,
                          log_probs: Any,
                          returns: Any,
                          advantage: Any) -> iter:

    batch_size = actions.size(0)
    for _ in range(batch_size // mini_batch_size):
      rand_ids = numpy.random.randint(0, batch_size, mini_batch_size)
      yield (states[rand_ids, :],
             actions[rand_ids, :],
             log_probs[rand_ids, :],
             returns[rand_ids, :],
             advantage[rand_ids, :])


def main():
  # test_agent_main(PPOAgent, C)
  _agent = PPOAgent()

  num_environments = 1
  import configs.agent_test_configs.test_ppo_config as C

  env_name = C.ENVIRONMENT_NAME

  def make_env(env_nam):
    @wraps(env_nam)
    def wrapper():
      env = gym.make(env_nam)
      return env

    return wrapper

  _environments = [make_env(env_name) for _ in range(num_environments)]
  _environments = U.SubprocVecEnv(_environments)

  _test_env = gym.make(env_name)

  _agent.input_size = _environments.observation_space.shape
  if len(_agent.input_size) == 0:
    _agent_input_size = (_environments.observation_space.n,)

  _agent.output_size = _environments.action_space.shape
  if len(_agent.output_size) == 0:
    _agent.output_size = (_environments.action_space.n,)

  _agent._maybe_infer_hidden_layers()

  # _agent.build(_test_env,device=_agent.device)

  _agent._actor_critic = U.ActorCritic(
      input_size=_agent._input_size,
      output_size=_agent._output_size,
      hidden_layers=_agent._hidden_layers
      ).to(_agent.device)

  _agent._actor_critic_target = U.ActorCritic(
      input_size=_agent._input_size,
      output_size=_agent._output_size,
      hidden_layers=_agent._hidden_layers
      ).to(_agent._device)

  _agent._actor_critic_target = U.copy_state(target=_agent._actor_critic_target, source=_agent._actor_critic)

  _agent._optimiser = optim.Adam(_agent._actor_critic.parameters(), lr=_agent._actor_critic_lr)

  _agent.train(_environments, _test_env)


if __name__ == '__main__':
  # main()

  import configs.agent_test_configs.test_ppo_config as C

  env_name = C.ENVIRONMENT_NAME
  _test_env = gym.make(env_name)

  test_agent_main(PPOAgent, C)
