#!/usr/local/bin/python
# coding: utf-8
from itertools import count
from typing import Any, Tuple

import numpy

from agent.architectures import DDPGActorArchitecture, DDPGCriticArchitecture
from agent.interfaces.partials.agents.torch_agents.actor_critic_agent import ActorCriticAgent
from agent.interfaces.specifications import ValuedTransition
from agent.interfaces.specifications.generalised_delayed_construction_specification import GDCS
from agent.training.procedures import batched_training
from agent.training.train_agent import parallelised_training, train_agent
from agent.utilities import to_tensor
from warg.named_ordered_dictionary import NOD

__author__ = 'cnheider'

import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

from agent import utilities as U


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
    self._current_kl_beta = 1.00

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

    self._actor_arch_spec = GDCS(DDPGActorArchitecture,
                                 kwargs=NOD({'input_shape':      None,  # Obtain from environment
                                             'hidden_layers':    None,
                                             'output_activation':None,
                                             'output_shape':     None,  # Obtain from environment
                                             }))

    self._critic_arch_spec = GDCS(DDPGCriticArchitecture,
                                  kwargs=NOD({'input_shape':      None,  # Obtain from environment
                                              'hidden_layers':    None,
                                              'output_activation':None,
                                              'output_shape':     None,  # Obtain from environment
                                              }))

    self._optimiser = None

    self._update_early_stopping = None
    # self._update_early_stopping = self.kl_target_stop

  def kl_target_stop(self,
                     old_log_probs,
                     new_log_probs,
                     kl_target=0.03,
                     beta_max=20,
                     beta_min=1 / 20):

    '''
    TRPO

    negloss = -tf.reduce_mean(self.advantages_ph * tf.exp(self.logp - self.prev_logp))
    negloss += tf.reduce_mean(self.beta_ph * self.kl_divergence)
    negloss += tf.reduce_mean(self.ksi_ph * tf.square(tf.maximum(0.0, self.kl_divergence - 2 *
    self.kl_target)))

    self.ksi = 10

    Adaptive kl_target = 0.01
    Adaptive kl_target = 0.03

    :param kl_target:
    :param beta_max:
    :param beta_min:
    :param old_log_probs:
    :param new_log_probs:
    :return:
    '''

    kl_now = torch.distributions.kl_divergence(old_log_probs, new_log_probs)
    if kl_now > 4 * kl_target:
      return True

    if kl_now < kl_target / 1.5:
      self._current_kl_beta /= 2
    elif kl_now > kl_target * 1.5:
      self._current_kl_beta *= 2
    self._current_kl_beta = numpy.clip(self._current_kl_beta, beta_min, beta_max)
    return False

  # endregion

  # region Protected

  def _optimise(self, cost, **kwargs):
    self._optimiser.zero_grad()

    cost.backward()

    if self._max_grad_norm is not None:
      nn.utils.clip_grad_norm(self._actor.parameters(), self._max_grad_norm)
      nn.utils.clip_grad_norm(self._critic.parameters(), self._max_grad_norm)

    self._optimiser.step()

  def _sample_model(self, state, *args, **kwargs):
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

    return action.detach().to('cpu').numpy(), action_log_prob, value_estimate, distribution

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
    T = tqdm(range(1, n + 1), f'Step #{self._step_i} - {0}/{n}', leave=False, disable=not render)
    for t in T:
      # T.set_description(f'Step #{self._step_i} - {t}/{n}')
      self._step_i += 1
      dist, value_estimates, *_ = self.sample_action(state)

      action = dist._sample()
      action_prob = dist.log_prob(action)

      next_state, signal, terminated, _ = environment.react(action)

      if render and self._rollout_i % render_frequency == 0:
        environment.render()

      successor_state = None
      if not terminated:  # If environment terminated then there is no successor state
        successor_state = next_state

      transitions.append(
          ValuedTransition(state,
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

  def react(self):
    pass



  def back_trace_advantages(self, transitions):
    n_step_summary = ValuedTransition(*zip(*transitions))

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
      step = ValuedTransition(*step)
      advantage_memories.append(
          ValuedTransition(
              step.state,
              step.action,
              discounted_returns[i],
              step.successor_state,
              step.terminal,
              step.action_prob,
              advantages[i]
              )
          )
      i += 1

    return advantage_memories

  def evaluate3(self, batch, discrete=False, **kwargs):
    # region Tensorise

    states = U.to_tensor(batch.state, device=self._device).view(-1, self._input_shape[0])

    value_estimates = U.to_tensor(batch.value_estimate, device=self._device)

    advantages = U.to_tensor(batch.advantage, device=self._device)

    discounted_returns = U.to_tensor(batch.discounted_return, device=self._device)

    action_probs_old = U.to_tensor(batch.action_prob, device=self._device).view(-1, self._output_shape[0])

    # endregion

    advantage = (advantages - advantages.mean()) / (advantages.std() + self._divide_by_zero_safety)

    *_, action_probs_new, distribution = self._sample_model(states)

    if discrete:
      actions = U.to_tensor(batch.action, device=self._device).view(-1, self._output_shape[0])
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

  def update_targets(self, *args, **kwargs) -> None:
    self.update_target(target_model=self._target_actor,
                       source_model=self._actor,
                       target_update_tau=self._target_update_tau)
    self.update_target(target_model=self._target_critic,
                       source_model=self._critic,
                       target_update_tau=self._target_update_tau)

  def evaluate(self, batch, _last_value_estimate, discrete=False, **kwargs):
    returns_ = U.compute_gae(_last_value_estimate,
                             batch.signal,
                             batch.non_terminal,
                             batch.value_estimate,
                             discount_factor=self._discount_factor,
                             tau=self._gae_tau)

    returns = torch.cat(returns_).detach()
    log_probs = torch.cat(batch.action_prob).detach()
    values = torch.cat(batch.value_estimate).detach()
    states = torch.cat(batch.state).view(-1, self._input_shape[0])
    actions = to_tensor(batch.action).view(-1,self._output_shape[0])

    advantage = returns - values

    self.inner_ppo_update(states,
                          actions,
                          log_probs,
                          returns,
                          advantage)

    if self._step_i % self._update_target_interval == 0:
      self.update_targets()

    return returns,log_probs,values,states,actions,advantage

  def update_models(self, *, stat_writer = None, **kwargs) -> None:
    pass
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
def ppo_test(rollouts=None, skip=True):
  import agent.configs.agent_test_configs.ppo_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  train_agent(PPOAgent,
              C,
              training_procedure=parallelised_training(training_procedure=batched_training,
                                                       auto_reset_on_terminal_state=True),
              parse_args=False,
              skip_confirmation=skip)


if __name__ == '__main__':
  ppo_test()
# endregion
