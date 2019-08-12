#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from neodroidagent.agents.model_free.hybrid.actor_critic_agent import ActorCriticAgent
from neodroidagent.architectures import SingleHeadMLP
from neodroidagent.architectures.experimental.merged import SingleHeadMergedInputMLP
from neodroidagent.interfaces.specifications import ArchitectureSpecification, GDCS
from neodroidagent.memory import TransitionBuffer
from neodroidagent.training.agent_session_entry_point import agent_session_entry_point
from neodroidagent.training.procedures import to_tensor, train_episodically
from neodroidagent.training.sessions.parallel_training import parallelised_training
from neodroidagent.utilities.exploration.sampling import OrnsteinUhlenbeckProcess
from draugr.writers.writer import Writer
from warg.named_ordered_dictionary import NOD

__author__ = 'cnheider'

tqdm.monitor_interval = 0


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
    self._random_process_spec = GDCS(constructor=OrnsteinUhlenbeckProcess,
                                     kwargs=dict(theta=0.15,
                                                 sigma=1.))

    # self._memory = U.PrioritisedReplayMemory(config.REPLAY_MEMORY_SIZE)  # Cuda trouble
    self._memory_buffer = TransitionBuffer(1000000)

    self._evaluation_function = F.smooth_l1_loss

    self._actor_arch_spec = ArchitectureSpecification(SingleHeadMLP,
                                                      kwargs=NOD(
                                                          {'input_shape':      None,
                                                           # Obtain from environment
                                                           'output_activation':torch.tanh,
                                                           'output_shape':     None,
                                                           # Obtain from environment
                                                           })
                                                      )

    self._critic_arch_spec = ArchitectureSpecification(SingleHeadMergedInputMLP,
                                                       kwargs=NOD(
                                                           {'input_shape': None,  # Obtain from environment
                                                            'output_shape':None,  # Obtain from environment
                                                            })
                                                       )

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

    self._batch_size = 64

    self._noise_factor = 3e-1
    self._low_action_clip = -1.0
    self._high_action_clip = 1.0

    (self._actor,
     self._target_actor,
     self._critic,
     self._target_critic,
     self._actor_optimiser,
     self._critic_optimiser) = None, None, None, None, None, None

    self._random_process = None

  # endregion

  # region Public

  def evaluate(self,
               batch
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
    states = to_tensor(state_batch,
                       device=self._device,
                       dtype=self._state_type)
    next_states = to_tensor(next_state_batch,
                            device=self._device,
                            dtype=self._state_type)
    actions = to_tensor(action_batch,
                        device=self._device,
                        dtype=self._action_type)
    signals = to_tensor(signal_batch,
                        device=self._device,
                        dtype=self._value_type)
    non_terminal_mask = to_tensor(non_terminal_batch,
                                  device=self._device,
                                  dtype=self._value_type)

    ### Critic ###
    # Compute current Q value, critic takes state and action chosen
    Q_current = self._critic(states, actions)
    # Compute next Q value based on which action target actor would choose
    # Detach variable from the current graph since we don't want gradients for next Q to propagated
    with torch.no_grad():
      target_actions = self._target_actor(states)
      next_max_q = self._target_critic(next_states, target_actions)

    next_Q_values = non_terminal_mask * next_max_q.view(next_max_q.shape[0], -1)

    Q_target = signals + (self._discount_factor * next_Q_values)  # Compute the target of the current Q values

    td_error = self._evaluation_function(Q_current.view(Q_current.shape[0], -1),
                                         Q_target)  # Compute Bellman error (using Huber loss)

    return td_error, states

  def update_targets(self):
    self._update_target(target_model=self._target_critic,
                        source_model=self._critic,
                        target_update_tau=self._target_update_tau)
    self._update_target(target_model=self._target_actor,
                        source_model=self._actor,
                        target_update_tau=self._target_update_tau)

  def update(self, *, stat_writer: Writer = None, **kwargs):
    '''
  Update the target networks

  :return:
  :rtype:
  '''
    if len(self._memory_buffer) < self._batch_size:
      return

    batch = self._memory_buffer.sample_transitions(self._batch_size)
    td_error, state_batch_var = self.evaluate(batch)
    critic_loss = self._optimise(temporal_difference_error=td_error,
                                 state_batch=state_batch_var)

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
                state_batch) -> float:
    '''

    :type kwargs: object
    '''
    self._optimise_critic(temporal_difference_error)

    ### Actor ###
    action_batch = self._actor(state_batch)
    c = self._critic(state_batch, action_batch)
    loss = -c.mean()
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
                    **kwargs):
    state = to_tensor(state, device=self._device, dtype=self._state_type)

    with torch.no_grad():
      action = self._actor(state)

    action_out = action.to('cpu').numpy()

    # Add action space noise for exploration, alternative is parameter space noise
    noise = self._random_process.sample(action_out.shape)
    action_out += noise * self._noise_factor

    if self._action_clipping:
      action_out = np.clip(action_out, self._low_action_clip, self._high_action_clip)

    return action_out

  # endregion

  def rollout(self,
              *args,
              **kwargs):
    self._random_process.reset()
    super().rollout(*args, **kwargs)


# region Test


def ddpg_test(rollouts=None, skip=True):
  import neodroidagent.configs.agent_test_configs.ddpg_test_config as C
  if rollouts:
    C.ROLLOUTS = rollouts

  agent_session_entry_point(DDPGAgent,
                            C,
                            training_session=parallelised_training(training_procedure=train_episodically,
                                                                   auto_reset_on_terminal_state=True),
                            parse_args=False,
                            skip_confirmation=skip)


def ddpg_run(rollouts=None, skip=True):
  import neodroidagent.configs.agent_test_configs.ddpg_test_config as C
  if rollouts:
    C.ROLLOUTS = rollouts

  agent_session_entry_point(DDPGAgent,
                            C,
                            training_session=parallelised_training(training_procedure=train_episodically,
                                                                   auto_reset_on_terminal_state=True),
                            parse_args=False,
                            skip_confirmation=skip)


if __name__ == '__main__':
  # ddpg_test()

  ddpg_run()
# endregion
