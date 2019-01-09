#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import draugr
from warg import NamedOrderedDictionary

from neodroid import EnvironmentState
from procedures.agent_tests import test_agent_main

__author__ = 'cnheider'
from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

tqdm.monitor_interval = 0

import utilities as U
from agents.abstract.ac_agent import ActorCriticAgent
from utilities.sampling.random_process import OrnsteinUhlenbeckProcess


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

  def __defaults__(self) -> None:
    # Adds noise for exploration
    self._random_process = OrnsteinUhlenbeckProcess(theta=0.15,
                                                    sigma=0.2)

    # self._memory = U.PrioritisedReplayMemory(config.REPLAY_MEMORY_SIZE)  # Cuda trouble
    self._memory_buffer = U.TransitionBuffer(1000000)

    self._evaluation_function = F.smooth_l1_loss

    self._actor_arch = U.DDPGActorArchitecture
    self._actor_arch_parameters = NamedOrderedDictionary({
      'input_size':       None,  # Obtain from environment
      'output_activation':torch.tanh,
      'output_size':      None,  # Obtain from environment
      })

    self._critic_arch = U.DDPGCriticArchitecture
    self._critic_arch_parameters = NamedOrderedDictionary({
      'input_size': None,  # Obtain from environment
      'output_size':None,  # Obtain from environment
      })

    self._discount_factor = 0.99

    self._initial_observation_period = 10000
    self._learning_frequency = 4
    self._sync_target_model_frequency = 10000
    self._state_type = torch.float
    self._value_type = torch.float
    self._action_type = torch.float

    self._epsilon_start = 0.9
    self._epsilon_end = 0.05
    self._epsilon_decay = 35000

    self._early_stopping_condition = None
    self._optimiser = None

    self._end_training = False

    self._batch_size = 64

    self._end_training = False

    (self._actor,
     self._target_actor,
     self._critic,
     self._target_critic,
     self._actor_optimiser,
     self._critic_optimiser) = None, None, None, None, None, None

    (self._input_size,
     self._output_size) = None, None

  def evaluate(self,
               batch,
               **kwargs
               ):
    '''

:type kwargs: object
'''
    (state_batch,
     action_batch,
     signal_batch,
     next_state_batch,
     non_terminal_batch) = batch
    states = U.to_tensor(state_batch, device=self._device, dtype=self._state_type) \
      .view(-1, self._input_size[0])
    next_states = U.to_tensor(next_state_batch, device=self._device, dtype=self._state_type) \
      .view(-1, self._input_size[0])
    actions = U.to_tensor(action_batch, device=self._device, dtype=self._action_type) \
      .view(-1, self._output_size[0])
    signals = U.to_tensor(signal_batch, device=self._device, dtype=self._value_type)

    non_terminal_mask = U.to_tensor(non_terminal_batch, device=self._device, dtype=self._value_type)

    ### Critic ###
    # Compute current Q value, critic takes state and action chosen
    Q_current = self._critic(states, actions=actions)
    # Compute next Q value based on which action target actor would choose
    # Detach variable from the current graph since we don't want gradients for next Q to propagated
    with torch.no_grad():
      target_actions = self._target_actor(states)
      next_max_q = self._target_critic(next_states, actions=target_actions).max(1)[0]

    next_Q_values = non_terminal_mask * next_max_q

    Q_target = signals + (self._discount_factor * next_Q_values)  # Compute the target of the current Q values

    td_error = self._evaluation_function(Q_current,
                                         Q_target.view(-1, 1))  # Compute Bellman error (using Huber loss)

    return td_error, states

  def update(self):
    '''
  Update the target networks

  :return:
  :rtype:
  '''
    if len(self._memory_buffer) < self._batch_size:
      return

    batch = self._memory_buffer.sample_transitions(self._batch_size)
    td_error, state_batch_var = self.evaluate(batch)
    loss = self._optimise_wrt_ac(td_error, state_batch_var)

    self.update_target(target_model=self._target_critic,
                       source_model=self._critic,
                       target_update_tau=self._target_update_tau)
    self.update_target(target_model=self._target_actor,
                       source_model=self._actor,
                       target_update_tau=self._target_update_tau)

    return td_error, loss

  def optimise_critic_wrt(self, error):
    self._critic_optimiser.zero_grad()
    error.backward()
    self._critic_optimiser.step()  # Optimize the critic

  def optimise_actor_wrt(self, loss):
    self._actor_optimiser.zero_grad()
    loss.backward()
    self._actor_optimiser.step()  # Optimize the actor

  def _optimise_wrt(self,
                    td_error,
                    **kwargs):
    pass

  def _optimise_wrt_ac(self,
                       td_error,
                       state_batch,
                       **kwargs):
    '''

    :type kwargs: object
    '''
    self.optimise_critic_wrt(td_error)

    ### Actor ###
    action_batch = self._actor(state_batch)
    loss = -self._critic(state_batch, actions=action_batch).mean()
    # loss = -torch.sum(self.critic(state_batch, self.actor(state_batch)))

    self.optimise_actor_wrt(loss)

    # self._memory.batch_update(indices, errors.tolist())  # Cuda trouble

    return loss

  def _sample_model(self,
                    state,
                    noise_factor=0.2,
                    low_action_clip=-1.0,
                    high_action_clip=1.0,
                    **kwargs):
    state = U.to_tensor([state], device=self._device, dtype=self._state_type)

    with torch.no_grad():
      action = self._actor(state)

    action_out = action.to('cpu').numpy()[0]
    #action_out = action.item()

    # Add action space noise for exploration, alternative is parameter space noise
    noise = self._random_process.sample()
    action_out += noise * noise_factor

    if self._action_clipping:
      action_out = np.clip(action_out, low_action_clip, high_action_clip)

    return action_out

  def _train(self,
             env,
             rollouts=1000,
             render=False,
             render_frequency=10,
             stat_frequency=10
             ):

    stats = draugr.StatisticCollection(stats=('signal', 'duration'))

    E = range(1, rollouts)
    E = tqdm(E, desc='', leave=False)

    for episode_i in E:
      state = env.reset()
      if type(state) is EnvironmentState:
        state = state.observables
      self._random_process.reset()

      if episode_i % stat_frequency == 0:
        draugr.styled_terminal_plot_stats_shared_x(stats, printer=E.write)
        E.set_description(f'Epi: {episode_i}, '
                          f'Sig: {stats.signal[-1]:.3f}'
                          )

      if render and episode_i % render_frequency == 0:
        signal, dur, *rollout_stats = self.rollout(state, env, render=render)
      else:
        signal, dur, *rollout_stats = self.rollout(state, env)

      stats.append(signal, dur)

      if self._end_training:
        break

    return (self._actor, self._critic), stats


def test_ddpg_agent(config):
  '''

:rtype: object
'''
  import gym

  device = torch.device('cuda' if config.USE_CUDA else 'cpu')

  env = gym.make(config.ENVIRONMENT_NAME)
  # env = NormaliseActionsWrapper(env)
  # env = neo.make('satellite',connect_to_running=False)

  agent = DDPGAgent(config)
  agent.build(env, device)
  listener = U.add_early_stopping_key_combination(agent.stop_training)

  listener.start()
  try:
    (actor_model, critic_model), stats = agent.train(env,
                                                     config.ROLLOUTS,
                                                     render=config.RENDER_ENVIRONMENT
                                                     )
  finally:
    listener.stop()

  U.save_model(actor_model, config, name='actor')
  U.save_model(critic_model, config, name='critic')


if __name__ == '__main__':
  import configs.agent_test_configs.test_ddpg_config as C

  test_agent_main(DDPGAgent, C)
