#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from agent.architectures import DDPGActorArchitecture, DDPGCriticArchitecture
from agent.exploration.sampling import OrnsteinUhlenbeckProcess
from agent.interfaces.partials.agents.torch_agents.actor_critic_agent import ActorCriticAgent
from agent.interfaces.specifications import ArchitectureSpecification
from agent.memory import TransitionBuffer
from agent.training.procedures import train_episodically
from agent.training.train_agent import parallelised_training, train_agent
from draugr.writers.writer import Writer
from warg.named_ordered_dictionary import NOD

__author__ = 'cnheider'

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

tqdm.monitor_interval = 0

from agent import utilities as U


class DDPGAgent(ActorCriticAgent):
  '''
  The Deep Deterministic Policy Gradient (DDPG) Agent

  Parameters
  ----------
      actor_optimizer_spec: OptimiserSpec
          Specifying the constructor and kwargs, as well as learning rate and other
          parameters for the optimiser
      critic_optimizer_spec: OptimiserSpec
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

  # region Private

  def __defaults__(self) -> None:
    # Adds noise for exploration
    self._random_process = OrnsteinUhlenbeckProcess(theta=0.15,
                                                    sigma=0.2)

    # self._memory = U.PrioritisedReplayMemory(config.REPLAY_MEMORY_SIZE)  # Cuda trouble
    self._memory_buffer = TransitionBuffer(1000000)

    self._evaluation_function = F.smooth_l1_loss

    self._actor_arch_spec = ArchitectureSpecification(DDPGActorArchitecture,
                                                      kwargs=NOD(
                                                          {'input_shape':      None,
                                                           # Obtain from environment
                                                           'output_activation':torch.tanh,
                                                           'output_shape':     None,
                                                           # Obtain from environment
                                                           }))

    self._critic_arch_spec = ArchitectureSpecification(DDPGCriticArchitecture,
                                                       kwargs=NOD(
                                                           {'input_shape': None,  # Obtain from environment
                                                            'output_shape':None,  # Obtain from environment
                                                            }))

    self._discount_factor = 0.95

    self._initial_observation_period = 10000
    self._learning_frequency = 4
    self._sync_target_model_frequency = 10000
    self._state_type = torch.float
    self._value_type = torch.float
    self._action_type = torch.float

    self._exploration_epsilon_start = 0.9
    self._exploration_epsilon_end = 0.05
    self._exploration_epsilon_decay = 10000

    self._early_stopping_condition = None
    self._optimiser = None

    self.end_training = False

    self._batch_size = 64

    self.end_training = False

    (self._actor,
     self._target_actor,
     self._critic,
     self._target_critic,
     self._actor_optimiser,
     self._critic_optimiser) = None, None, None, None, None, None

    (self._input_shape,
     self._output_shape) = None, None

  # endregion

  # region Public

  def evaluate(self,
               batch,
               **kwargs
               ):
    '''

    :param batch:
:type kwargs: object
'''
    (state_batch,
     action_batch,
     signal_batch,
     next_state_batch,
     non_terminal_batch) = batch
    states = U.to_tensor(state_batch,
                         device=self._device,
                         dtype=self._state_type).view(-1, self._input_shape[0])
    next_states = U.to_tensor(next_state_batch,
                              device=self._device,
                              dtype=self._state_type).view(-1, self._input_shape[0])
    actions = U.to_tensor(action_batch,
                          device=self._device,
                          dtype=self._action_type).view(-1, self._output_shape[0])
    signals = U.to_tensor(signal_batch,
                          device=self._device,
                          dtype=self._value_type).view(-1, 1)
    non_terminal_mask = U.to_tensor(non_terminal_batch,
                                    device=self._device,
                                    dtype=self._value_type).view(-1, 1)

    ### Critic ###
    # Compute current Q value, critic takes state and action chosen
    Q_current = self._critic(states, actions=actions)
    # Compute next Q value based on which action target actor would choose
    # Detach variable from the current graph since we don't want gradients for next Q to propagated
    with torch.no_grad():
      target_actions = self._target_actor(states)
      next_max_q = self._target_critic(next_states, actions=target_actions)

    next_Q_values = non_terminal_mask * next_max_q

    Q_target = signals + (self._discount_factor * next_Q_values)  # Compute the target of the current Q values

    td_error = self._evaluation_function(Q_current,
                                         Q_target.view(-1, 1))  # Compute Bellman error (using Huber loss)

    return td_error, states

  def update_targets(self):
    self.update_target(target_model=self._target_critic,
                       source_model=self._critic,
                       target_update_tau=self._target_update_tau)
    self.update_target(target_model=self._target_actor,
                       source_model=self._actor,
                       target_update_tau=self._target_update_tau)

  def update_models(self, *, stat_writer: Writer = None, **kwargs):
    '''
  Update the target networks

  :return:
  :rtype:
  '''
    if len(self._memory_buffer) < self._batch_size:
      return

    batch = self._memory_buffer.sample_transitions(self._batch_size)
    td_error, state_batch_var = self.evaluate(batch)
    critic_loss = self._optimise(temporal_difference_error=td_error, state_batch=state_batch_var)

    self.update_targets()

    if stat_writer:
      stat_writer.scalar('td_error', td_error)
      stat_writer.scalar('critic_loss', critic_loss)

    return td_error, critic_loss

  # endregion

  # region Protected

  def _optimise(self,
                *,
                temporal_difference_error,
                state_batch,
                **kwargs) -> float:
    '''

    :type kwargs: object
    '''
    self._optimise_critic(temporal_difference_error)

    ### Actor ###
    action_batch = self._actor(state_batch)
    loss = -self._critic(state_batch, actions=action_batch).mean()
    # loss = -torch.sum(self.critic(state_batch, self.actor(state_batch)))

    self._optimise_actor(loss)

    # self._memory.batch_update(indices, errors.tolist())  # Cuda trouble

    return loss

  def _optimise_critic(self, error):
    self._critic_optimiser.zero_grad()
    error.backward()
    self._critic_optimiser.step()  # Optimize the critic

  def _optimise_actor(self, loss):
    self._actor_optimiser.zero_grad()
    loss.backward()
    self._actor_optimiser.step()  # Optimize the actor

  def _sample_model(self,
                    state,
                    *,
                    noise_factor=0.2,
                    low_action_clip=-1.0,
                    high_action_clip=1.0,
                    **kwargs):
    state = U.to_tensor(state, device=self._device, dtype=self._state_type)

    with torch.no_grad():
      action = self._actor(state)

    action_out = action.to('cpu').numpy()

    # Add action space noise for exploration, alternative is parameter space noise
    noise = self._random_process.sample()
    action_out += noise * noise_factor

    if self._action_clipping:
      action_out = np.clip(action_out, low_action_clip, high_action_clip)

    return action_out

  # endregion

  def rollout(self,
              initial_state,
              environment,
              render=False,
              train=True,
              **kwargs):
    self._random_process.reset()
    super().rollout(initial_state, environment, render, train)


# region Test


def ddpg_test(rollouts=None, skip=True):
  import agent.configs.agent_test_configs.ddpg_test_config as C
  if rollouts:
    C.ROLLOUTS = rollouts

  train_agent(DDPGAgent,
              C,
              training_session=parallelised_training(training_procedure=train_episodically,
                                                     auto_reset_on_terminal_state=True),
              parse_args=False,
              skip_confirmation=skip)


if __name__ == '__main__':
  ddpg_test()
# endregion
