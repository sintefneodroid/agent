#!/usr/local/bin/python
# coding: utf-8
from itertools import count
from typing import Any, Tuple

import numpy

from agent.architectures import DDPGActorArchitecture, DDPGCriticArchitecture
from agent.procedures.train_agent import agent_test_main, parallel_train_agent_procedure
from agent.utilities.specifications.generalised_delayed_construction_specification import GDCS
from agent.utilities.specifications.training_resume import TR
from warg import NOD

__author__ = 'cnheider'

import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

from agent import utilities as U
from agent.agents.abstract.ac_agent import ActorCriticAgent


class PPOAgent(ActorCriticAgent):
  '''
  PPO, Proximal Policy Optimization method

  See method __defaults__ for default parameters
'''

  # region Private

  def __defaults__(self) -> None:
    self._steps = 10

    self._discount_factor = 0.99
    self._gae_tau = 0.95
    # self._reached_horizon_penalty = -10.

    self._actor_lr = 3e-4
    self._critic_lr = 3e-3
    self._entropy_reg_coef = 3e-3
    self._value_reg_coef = 5e-1
    self._batch_size = 64
    self._mini_batch_size = 10
    self._initial_observation_period = 0
    self._target_update_tau = 1.0
    self._update_target_interval = 1000
    self._max_grad_norm = None
    self._solved_threshold = -200
    self._test_interval = 1000
    self._early_stop = False
    self._rollouts = 10000

    self._ppo_epochs = 4

    self._state_type = torch.float
    self._value_type = torch.float
    self._action_type = torch.long

    # params for epsilon greedy
    self._exploration_epsilon_start = 0.99
    self._exploration_epsilon_end = 0.05
    self._exploration_epsilon_decay = 10000

    self._use_cuda = False

    self._surrogate_clipping_value = 0.2

    self._optimiser_spec = GDCS(torch.optim.Adam, {})

    self._actor_arch_spec = GDCS(DDPGActorArchitecture, kwargs=NOD({
      'input_size':       None,  # Obtain from environment
      'hidden_layers':    None,
      'output_activation':None,
      'output_size':      None,  # Obtain from environment
      }))

    self._critic_arch_spec = GDCS(DDPGCriticArchitecture, kwargs=NOD({
      'input_size':       None,  # Obtain from environment
      'hidden_layers':    None,
      'output_activation':None,
      'output_size':      None,  # Obtain from environment
      }))

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

  def _train_procedure(self, *args, **kwargs):

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

      next_state, signal, terminated, _ = environment.react(action)

      if render and self._rollout_i % render_frequency == 0:
        environment.render()

      successor_state = None
      if not terminated:  # If environment terminated then there is no successor state
        successor_state = next_state

      transitions.append(
          U.ValuedTransition(state,
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
              render=False,
              train=True,
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

      next_state, signal, terminated, _ = environment.react(action)

      if render:
        environment.render()

      successor_state = None
      if not terminated:  # If environment terminated then there is no successor state
        successor_state = next_state

      transitions.append(
          U.ValuedTransition(state,
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

  def back_trace_advantages(self, transitions):
    n_step_summary = U.ValuedTransition(*zip(*transitions))

    advantages = U.advantage_estimate(n_step_summary.signal,
                                      n_step_summary.non_terminal,
                                      n_step_summary.value_estimate,
                                      discount_factor=self._discount_factor,
                                      tau=self._gae_tau,
                                      device=self._device
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

    action_probs_old = U.to_tensor(batch.action_prob, device=self._device).view(-1, self._output_size[0])

    # endregion

    advantage = (advantages - advantages.mean()) / (advantages.std() + self._divide_by_zero_safety)

    *_, action_probs_new, distribution = self._sample_model(states)

    if discrete:
      actions = U.to_tensor(batch.action, device=self._device).view(-1, self._output_size[0])
      action_probs_old = action_probs_old.gather(1, actions)
      action_probs_new = action_probs_new.gather(1, actions)

    ratio = (action_probs_new - action_probs_old).exp()
    # Generated action probs from (new policy) and (old policy).
    # Values of [0..1] means that actions less likely with the new policy,
    # while values [>1] mean action a more likely now

    surrogate = ratio * advantage

    clamped_ratio = torch.clamp(ratio,
                                min=1. - self._surrogate_clipping_value,
                                max=1. + self._surrogate_clipping_value)
    surrogate_clipped = clamped_ratio * advantage  # (L^CLIP)

    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

    entropy_loss = distribution.entropy().mean()

    # value_error = (value_estimates - discounted_returns).pow(2).mean()
    value_error = F.mse_loss(value_estimates, discounted_returns)

    collective_cost = policy_loss + value_error * self._value_reg_coef - entropy_loss * self._entropy_reg_coef

    return collective_cost, policy_loss, value_error

  def evaluate2(self,
                *,
                states,
                actions,
                log_probs,
                returns,
                advantage,
                **kwargs):
    action_out, action_log_prob, value_estimate, distribution = self._sample_model(states)

    old_log_probs = log_probs
    new_log_probs = distribution.log_prob(actions)

    ratio = (new_log_probs - old_log_probs).exp()
    surrogate = ratio * advantage
    surrogate_clipped = (torch.clamp(ratio,
                                     1.0 - self._surrogate_clipping_value,
                                     1.0 + self._surrogate_clipping_value)
                         * advantage)

    actor_loss = - torch.min(surrogate, surrogate_clipped).mean()
    # critic_loss = (value_estimate-returns).pow(2).mean()
    critic_loss = F.mse_loss(value_estimate, returns)

    entropy = distribution.entropy().mean()

    loss = self._value_reg_coef * critic_loss + actor_loss - entropy + self._entropy_reg_coef
    return loss, new_log_probs, old_log_probs

  def update(self):

    returns_ = U.compute_gae(self._last_value_estimate,
                             self._transitions_.signal,
                             self._transitions_.non_terminal,
                             self._transitions_.value_estimate,
                             discount_factor=self._discount_factor,
                             tau=self._gae_tau)

    returns = torch.cat(returns_).view(-1, 1).detach()
    log_probs = torch.cat(self._transitions_.action_prob).detach()
    values = torch.cat(self._transitions_.value_estimate).detach()
    states = torch.cat(self._transitions_.state).view(-1, self._input_size[0])
    actions = torch.cat(self._transitions_.action)

    advantage = returns - values

    self.inner_ppo_update(states,
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

  def inner_ppo_update(self,
                       states,
                       actions,
                       log_probs,
                       returns,
                       advantages,
                       ):
    mini_batch_gen = self.ppo_mini_batch_iter(self._mini_batch_size,
                                              states,
                                              actions,
                                              log_probs,
                                              returns,
                                              advantages)
    for _ in range(self._ppo_epochs):
      try:
        batch = mini_batch_gen.__next__()
      except StopIteration:
        return

      loss, new_log_probs, old_log_probs = self.evaluate2(**batch.as_dict())

      self._actor_optimiser.zero_grad()
      self._critic_optimiser.zero_grad()
      loss.backward()
      self._actor_optimiser.step()
      self._critic_optimiser.step()

      if self._update_early_stopping:
        if self._update_early_stopping(old_log_probs, new_log_probs):
          break

  def sample_action(self, state, **kwargs) -> Tuple[Any, Any, Any]:
    action, action_log_prob, value_estimate, *_ = self._sample_model(state)

    return action, action_log_prob, value_estimate

  def train_episodically(self,
                         env,
                         rollouts=10000,
                         render=False,
                         render_frequency=1000,
                         stat_frequency=10,
                         **kwargs):

    self._rollout_i = 1

    initial_state = env.reset()

    B = tqdm(range(1, rollouts + 1), f'Batch {0}, {rollouts} - Rollout {self._rollout_i}', leave=False)
    for batch_i in B:
      if self._end_training or batch_i > rollouts:
        break

      if batch_i % stat_frequency == 0:
        B.set_description(f'Batch {batch_i}, {rollouts} - Rollout {self._rollout_i}')

      if render and batch_i % render_frequency == 0:
        transitions, accumulated_signal, terminated, *_ = self.rollout(initial_state,
                                                                       env,
                                                                       render=render
                                                                       )
      else:
        transitions, accumulated_signal, terminated, *_ = self.rollout(initial_state,
                                                                       env
                                                                       )

      initial_state = env.reset()

      if batch_i >= self._initial_observation_period:
        advantage_memories = self.back_trace_advantages(transitions)

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

  def train_step_batched(self,
                         environments,
                         test_environments,
                         *,
                         num_steps=200,
                         rollouts=10000,
                         render=True
                         ) -> TR:
    # stats = draugr.StatisticCollection(stats=('batch_signal', 'test_signal', 'entropy'))

    state = environments.reset()

    S = tqdm(range(rollouts), leave=False)
    for i in S:
      S.set_description(f'Rollout {i}')
      if self._end_training:
        break

      batch_signal = []

      transitions = []

      successor_state = None

      state = U.to_tensor(state, device=self._device)

      I = tqdm(range(num_steps), leave=False)
      for _ in I:

        action, action_log_prob, value_estimate = self.sample_action(state)

        a = action.to('cpu').numpy()
        successor_state, signal, terminated, *_ = environments.react(a)

        batch_signal.append(signal)

        successor_state = U.to_tensor(successor_state, device=self._device)
        signal_ = U.to_tensor(signal, device=self._device)
        not_terminated = U.to_tensor([not t for t in terminated], device=self._device)

        transitions.append(U.ValuedTransition(state,
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

        if self._step_i % self._test_interval == 0 and test_environments:
          test_signals = self.test_agent(test_environments, render=render)
          test_signal = test_signals
          # test_signal = numpy.mean(test_signals)
          # stats.test_signal.append(test_signal)

          # draugr.styled_terminal_plot_stats_shared_x(stats, printer=S.write)

          if test_signal > self._solved_threshold and self._early_stop:
            self._end_training = True

        # stats.batch_signal.append(batch_signal)

      # only calculate value of next state for the last step this time
      *_, self._last_value_estimate, _ = self._sample_model(successor_state)

      self._transitions_ = U.ValuedTransition(*zip(*transitions))

      if len(self._transitions_) > 100:
        self.update()

    return TR((self._actor, self._critic), None)

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
    out_signal = 0
    while not terminal:
      state = U.to_tensor(state, device=self._device)
      with torch.no_grad():
        action, *_ = self.sample_action(state)
        next_state, signal, terminal = test_environment.react(action)

      # terminal = terminal.all()
      state = next_state
      if render:
        test_environment.render()

      out_signal = signal
      # out_signal += signal.mean()
    return out_signal

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
      yield NOD(states=states[rand_ids, :],
                actions=actions[rand_ids, :],
                log_probs=log_probs[rand_ids, :],
                returns=returns[rand_ids, :],
                advantage=advantage[rand_ids, :])

  # endregion


# region Test
def ppo_test(rollouts=None):
  import agent.configs.agent_test_configs.ppo_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  agent_test_main(PPOAgent,
                  C,
                  training_procedure=parallel_train_agent_procedure(
                      auto_reset_on_terminal_state=True),
                  parse_args=False)


if __name__ == '__main__':
  ppo_test()
# endregion
