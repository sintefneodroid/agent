#!/usr/local/bin/python
# coding: utf-8

import numpy

from agent.architectures import DDPGActorArchitecture, DDPGCriticArchitecture, ContinuousActorArchitecture
from agent.interfaces.partials.agents.torch_agents.actor_critic_agent import ActorCriticAgent
from agent.interfaces.specifications import AdvantageDiscountedTransition, ValuedTransition
from agent.interfaces.specifications.generalised_delayed_construction_specification import GDCS
from agent.training.procedures import batched_training
from agent.training.train_agent import parallelised_training, train_agent
from agent.utilities import to_tensor
from warg.named_ordered_dictionary import NOD

__author__ = 'cnheider'

import torch
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

    self._optimiser_spec = GDCS(torch.optim.Adam, {})

    self._actor_arch_spec = GDCS(ContinuousActorArchitecture,
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

  def tpro_kl_target_stop(self,
                          old_log_probs,
                          new_log_probs,
                          kl_target=3e-2,

                          kl_coef_min=1e-4,
                          kl_coef_max=10):

    '''

    :param kl_target:
    :param kl_coef_max:
    :param kl_coef_min:
    :param old_log_probs:
    :param new_log_probs:
    :return:
    '''

    # kl_divergence = torch.distributions.kl_divergence(old_log_probs, new_log_probs)
    kl_divergence = (old_log_probs - new_log_probs).mean()
    if kl_divergence > 4 * kl_target:
      return True

    if kl_divergence < kl_target / 1.5:
      self._kl_reg_coef /= 2
    elif kl_divergence > kl_target * 1.5:
      self._kl_reg_coef *= 2
    self._kl_reg_coef = numpy.clip(self._kl_reg_coef, kl_coef_min, kl_coef_max)

    return False

  # endregion

  # region Protected

  def _optimise(self, cost, **kwargs):

    self._actor_optimiser.zero_grad()
    self._critic_optimiser.zero_grad()
    cost.backward(retain_graph=True)
    self._actor_optimiser.step()
    self._critic_optimiser.step()

    '''
    self._optimiser.zero_grad()

    cost.backward()

    if self._max_grad_norm is not None:
      nn.utils.clip_grad_norm(self._actor.parameters(), self._max_grad_norm)
      nn.utils.clip_grad_norm(self._critic.parameters(), self._max_grad_norm)

    self._optimiser.step()
    '''

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


    distribution = torch.distributions.Normal(mean, std)

    with torch.no_grad():
      action = distribution.sample()

    value_estimate = self._critic(model_input,actions=action)

    action_log_prob = distribution.log_prob(action)

    return action.detach().to('cpu').numpy(), action_log_prob, value_estimate, distribution

  # endregion

  # region Public

  def take_n_steps(self,
                   initial_state,
                   environment,
                   n=100,
                   render=False):
    state = initial_state

    accumulated_signal = 0

    transitions = []
    terminated = False
    T = tqdm(range(1, n + 1), f'Step #{self._step_i} - {0}/{n}', leave=False, disable=not render)
    for t in T:
      # T.set_description(f'Step #{self._step_i} - {t}/{n}')
      self._step_i += 1
      action, action_prob, value_estimates, *_ = self.sample_action(state)

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

    advantages = U.advantage_estimate(transitions.signal,
                                      transitions.non_terminal,
                                      transitions.value_estimate,
                                      discount_factor=self._discount_factor,
                                      tau=self._gae_tau,
                                      device=self._device
                                      )

    value_estimates = U.to_tensor(transitions.value_estimate, device=self._device)

    discounted_returns = value_estimates + advantages

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
                                                              discounted_returns[i],
                                                              advantages[i]
                                                              )
                                )
      i += 1

    return AdvantageDiscountedTransition(*zip(*advantage_memories))

  def evaluate(self, batch: AdvantageDiscountedTransition, discrete=False, **kwargs):
    # region Tensorise

    states = to_tensor(batch.state, device=self._device)
    value_estimates = to_tensor(batch.value_estimate, device=self._device)
    advantages = to_tensor(batch.advantage, device=self._device)
    discounted_returns = to_tensor(batch.discounted_return, device=self._device)
    action_log_probs_old = to_tensor(batch.action_prob, device=self._device)

    # endregion

    advantage = (advantages - advantages.mean()) / (advantages.std() + self._divide_by_zero_safety)

    *_, action_log_probs_new, distribution = self._sample_model(states)

    if discrete:
      actions = U.to_tensor(batch.action, device=self._device).view(-1, self._output_shape[0])
      action_log_probs_old = action_log_probs_old.gather(1, actions)
      action_log_probs_new = action_log_probs_new.gather(1, actions)

    ratio = (action_log_probs_new - action_log_probs_old).exp()
    # Generated action probs from (new policy) and (old policy).
    # Values of [0..1] means that actions less likely with the new policy,
    # while values [>1] mean action a more likely now
    surrogate = ratio * advantage
    clamped_ratio = torch.clamp(ratio,
                                min=1. - self._surrogate_clipping_value,
                                max=1. + self._surrogate_clipping_value)
    surrogate_clipped = clamped_ratio * advantage  # (L^CLIP)

    policy_loss = torch.min(surrogate, surrogate_clipped).mean()
    entropy_loss = distribution.entropy().mean() * self._entropy_reg_coef
    policy_loss -= entropy_loss

    value_error = F.mse_loss(value_estimates, discounted_returns) * self._value_reg_coef
    collective_cost = policy_loss + value_error

    return collective_cost, policy_loss, value_error,

  def update_targets(self) -> None:
    self.update_target(target_model=self._target_actor,
                       source_model=self._actor,
                       target_update_tau=self._target_update_tau)

    self.update_target(target_model=self._target_critic,
                       source_model=self._critic,
                       target_update_tau=self._target_update_tau)

  def update_models(self, *, stat_writer=None, **kwargs) -> None:

    self.ppo_updates(self.transitions)

    if self._step_i % self._update_target_interval == 0:
      self.update_targets()

  def ppo_updates(self,
                  batch,
                  mini_batches=16
                  ):
    batch_size = len(batch) // mini_batches
    # mini_batch_generator = DataLoader(batch, batch_size=batch_size, shuffle=True) #pin_memory=True
    mini_batch_generator = self.ppo_mini_batch_iter(batch_size, batch)
    for i in range(self._ppo_epochs):
      for mini_batch in mini_batch_generator:
        mini_batch_adv = self.back_trace_advantages(mini_batch)
        loss, new_log_probs, old_log_probs = self.evaluate(mini_batch_adv)

        self._optimise(loss)

        # if self.tpro_kl_target_stop(old_log_probs, new_log_probs):
        #  logging.info(f'Early stopping at update {i} due to reaching max kl.')
        #  break

  # endregion

  @staticmethod
  def ppo_mini_batch_iter(mini_batch_size: int,
                          batch: ValuedTransition) -> iter:

    batch_size = len(batch)
    for _ in range(batch_size // mini_batch_size):
      rand_ids = numpy.random.randint(0, batch_size, mini_batch_size)
      a = batch[:, rand_ids]
      yield ValuedTransition(*a)


# region Test
def ppo_test(rollouts=None, skip=True):
  import agent.configs.agent_test_configs.ppo_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  train_agent(PPOAgent,
              C,
              training_session=parallelised_training(training_procedure=batched_training,
                                                     auto_reset_on_terminal_state=True),
              parse_args=False,
              skip_confirmation=skip)


if __name__ == '__main__':
  ppo_test()
# endregion
