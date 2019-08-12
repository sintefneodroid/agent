#!/usr/local/bin/python
# coding: utf-8
from typing import Any

import numpy
from torch import nn

from neodroid.interfaces.specifications import EnvironmentSnapshot
from neodroidagent.interfaces.specifications import AdvantageDiscountedTransition, ValuedTransition
from neodroidagent.training.agent_session_entry_point import agent_session_entry_point
from neodroidagent.training.procedures import  step_wise_training, to_tensor
from neodroidagent.training.sessions.parallel_training import parallelised_training
from neodroidagent.utilities.signal.advantage_estimation import torch_compute_gae
from neodroidagent.utilities.signal.experimental.discounting import discount_signal
from draugr.writers.writer import Writer
from neodroid.environments.vector_environment import VectorEnvironment
from .actor_critic_agent import ActorCriticAgent

__author__ = 'cnheider'

import torch
from tqdm import tqdm
import torch.nn.functional as F


class PPOAgent(ActorCriticAgent):
  '''
  PPO, Proximal Policy Optimization method

  See method __defaults__ for default parameters
'''

  # region Private

  def __defaults__(self) -> None:
    self._steps = 10

    self._discount_factor = 0.95
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

    (self._actor,
     self._target_actor,
     self._critic,
     self._target_critic,
     self._actor_optimiser,
     self._critic_optimiser) = None, None, None, None, None, None

  # endregion

  # region Protected

  def _optimise(self, cost, **kwargs):

    self._actor_optimiser.zero_grad()
    self._critic_optimiser.zero_grad()
    cost.backward(retain_graph=True)

    if self._max_grad_norm is not None:
      nn.utils.clip_grad_norm(self._actor.parameters(), self._max_grad_norm)
      nn.utils.clip_grad_norm(self._critic.parameters(), self._max_grad_norm)

    self._actor_optimiser.step()
    self._critic_optimiser.step()

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

    model_input = to_tensor(state, device=self._device, dtype=self._state_type)

    distribution = self._actor(model_input)[0]

    with torch.no_grad():
      action = distribution.sample()

    value_estimate = self._critic(model_input, action)[0]

    action_log_prob = distribution.log_prob(action)

    return (action.detach().to('cpu').numpy(),
            action_log_prob,
            value_estimate,
            distribution)

  def _ppo_updates(self,
                   batch,
                   mini_batches=16
                   ):
    batch_size = len(batch) // mini_batches
    mini_batch_generator = self._ppo_mini_batch_iter(batch_size, batch)
    for i in range(self._ppo_epochs):
      for mini_batch in mini_batch_generator:
        mini_batch_adv = self.back_trace_advantages(mini_batch)
        loss, new_log_probs, old_log_probs = self.evaluate(mini_batch_adv)

        self._optimise(loss)

  @staticmethod
  def _ppo_mini_batch_iter(mini_batch_size: int,
                           batch: ValuedTransition) -> iter:

    batch_size = len(batch)
    for _ in range(batch_size // mini_batch_size):
      rand_ids = numpy.random.randint(0, batch_size, mini_batch_size)
      a = batch[:, rand_ids]
      yield ValuedTransition(*a)

  def _update_targets(self) -> None:
    self._update_target(target_model=self._target_actor,
                        source_model=self._actor,
                        target_update_tau=self._target_update_tau)

    self._update_target(target_model=self._target_critic,
                        source_model=self._critic,
                        target_update_tau=self._target_update_tau)

  # endregion

  # region Public

  def take_n_steps(self,
                   initial_state: EnvironmentSnapshot,
                   environment: VectorEnvironment,
                   n: int = 100,
                   *,
                   train: bool = False,
                   render: bool = False,
                   **kwargs) -> Any:
    state = initial_state

    accumulated_signal = 0

    transitions = []
    terminated = False
    T = tqdm(range(1, n + 1),
             f'Step #{self._step_i} - {0}/{n}',
             leave=False,
             disable=not render)
    for t in T:
      # T.set_description(f'Step #{self._step_i} - {t}/{n}')
      self._step_i += 1
      action, action_prob, value_estimates, *_ = self.sample(state)

      next_state, signal, terminated, _ = environment.react(action)

      if render:
        environment.render()

      successor_state = None
      if not terminated:  # If environment terminated then there is no successor state
        successor_state = next_state

      transitions.append(ValuedTransition(state,
                                          action,
                                          action_prob,
                                          value_estimates,
                                          signal,
                                          successor_state,
                                          terminated,
                                          )
                         )

      state = next_state

      accumulated_signal += signal

      if terminated:
        state = environment.reset()

    self.transitions = transitions

    return transitions, accumulated_signal, terminated, state

  def back_trace_advantages(self, transitions):

    value_estimates = to_tensor(transitions.value_estimate, device=self._device)
    sig = to_tensor(transitions.signal)
    value_estimates = value_estimates.view(value_estimates.shape[0], -1)

    advantages = torch_compute_gae(signals=sig,
                                   values=value_estimates,
                                   non_terminals=transitions.non_terminal,
                                   discount_factor=self._discount_factor,
                                   tau=self._gae_tau
                                   )

    discounted_signal = discount_signal(sig.transpose(0, 1).detach().to('cpu').numpy(),
                                        self._discount_factor).transpose()

    i = 0
    advantage_memories = []
    for step in zip(*transitions):
      step = ValuedTransition(*step)
      advantage_memories.append(AdvantageDiscountedTransition(step.state,
                                                              step.action,
                                                              step.successor_state,
                                                              step.terminal,
                                                              step.action_prob,
                                                              value_estimates,
                                                              discounted_signal[i],
                                                              advantages[i]
                                                              )
                                )
      i += 1

    return AdvantageDiscountedTransition(*zip(*advantage_memories))

  def evaluate(self,
               batch: AdvantageDiscountedTransition,
               discrete: bool = False,
               **kwargs):
    # region Tensorise

    states = to_tensor(batch.state, device=self._device)
    value_estimates = to_tensor(batch.value_estimate, device=self._device)
    advantages = to_tensor(batch.advantage, device=self._device)
    discounted_returns = to_tensor(batch.discounted_return, device=self._device)
    action_log_probs_old = to_tensor(batch.action_prob, device=self._device)

    # endregion

    *_, action_log_probs_new, distribution = self._sample_model(states)

    if discrete:
      actions = to_tensor(batch.action, device=self._device)
      action_log_probs_old = action_log_probs_old.gather(-1, actions)
      action_log_probs_new = action_log_probs_new.gather(-1, actions)

    ratio = (action_log_probs_new - action_log_probs_old).exp()
    # Generated action probs from (new policy) and (old policy).
    # Values of [0..1] means that actions less likely with the new policy,
    # while values [>1] mean action a more likely now
    surrogate = ratio * advantages
    clamped_ratio = torch.clamp(ratio,
                                min=1. - self._surrogate_clipping_value,
                                max=1. + self._surrogate_clipping_value)
    surrogate_clipped = clamped_ratio * advantages  # (L^CLIP)

    policy_loss = torch.min(surrogate, surrogate_clipped).mean()
    entropy_loss = distribution.entropy().mean() * self._entropy_reg_coef
    policy_loss -= entropy_loss

    value_error = F.mse_loss(value_estimates, discounted_returns) * self._value_reg_coef
    collective_cost = policy_loss + value_error

    return collective_cost, policy_loss, value_error,

  def update(self, *, stat_writer: Writer = None, **kwargs) -> None:

    self._ppo_updates(self.transitions)

    if self._step_i % self._update_target_interval == 0:
      self._update_targets()

  # endregion


# region Test
def ppo_test(rollouts=None, skip: bool = True):
  import neodroidagent.configs.agent_test_configs.ppo_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  agent_session_entry_point(PPOAgent,
                            C,
                            training_session=parallelised_training(training_procedure=step_wise_training,
                                                                   auto_reset_on_terminal_state=True),
                            parse_args=False,
                            skip_confirmation=skip)


def ppo_run(rollouts=None, skip: bool = True):
  import neodroidagent.configs.agent_test_configs.ppo_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  agent_session_entry_point(PPOAgent,
                            C,
                            training_session=parallelised_training(training_procedure=step_wise_training,
                                                                   auto_reset_on_terminal_state=True),
                            parse_args=False,
                            skip_confirmation=skip)


if __name__ == '__main__':
  # ppo_test()
  ppo_run()

# endregion
