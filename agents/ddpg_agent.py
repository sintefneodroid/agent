#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import draugr
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
from agents.abstract.joint_ac_agent import JointACAgent
from utilities.sampling.random_process import OrnsteinUhlenbeckProcess


class DDPGAgent(JointACAgent):
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
    self._optimiser_type = torch.optim.Adam

    self._actor_optimiser_spec = U.OptimiserSpecification(constructor=self._optimiser_type,
                                                          kwargs=dict(lr=0.0001)
                                                          )
    self._critic_optimiser_spec = U.OptimiserSpecification(constructor=self._optimiser_type,
                                                           kwargs=dict(lr=0.001,
                                                                       weight_decay=0.01)
                                                           )

    self._random_process = OrnsteinUhlenbeckProcess(theta=0.15,
                                                    sigma=0.2)
    # Adds noise for exploration

    # self._memory = U.PrioritisedReplayMemory(config.REPLAY_MEMORY_SIZE)  # Cuda trouble
    self._memory = U.TransitionBuffer(1000000)
    self._evaluation_function = F.smooth_l1_loss

    self._actor_arch = U.ActorArchitecture
    self._actor_arch_parameters = {
      'input_size':       None,  # Obtain from environment
      'hidden_layers':    [128, 64],
      'output_activation':None,
      'output_size':      None,  # Obtain from environment
      }

    self._critic_arch = U.CriticArchitecture
    self._critic_arch_parameters = {
      'input_size':       None,  # Obtain from environment
      'hidden_layers':    [128, 64],
      'output_activation':None,
      'output_size':      None,  # Obtain from environment
      }

    self._discount_factor = 0.99
    self._use_double_dqn = False
    self._signal_clipping = False
    self._action_clipping = False
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

    self._batch_size = 60
    self._target_update_tau = 0.001

    self._end_training = False

    (self._actor,
     self._target_actor,
     self._critic,
     self._target_critic,
     self._actor_optimiser,
     self._critic_optimiser) = None, None, None, None, None, None

    (self._input_size,
     self._output_size) = None, None

  def save(self, C):
    U.save_model(self._actor, C, name='actor')
    U.save_model(self._critic, C, name='policy')

  def load(self,
           model_path,
           evaluation=False,
           **kwargs):
    print('loading latest model: ' + model_path)

    self._build(**kwargs)

    self._actor.load_state_dict(torch.load(f'actor-{model_path}'))
    self._critic.load_state_dict(torch.load(f'critic-{model_path}'))

    self.update_target(self._target_critic, self._critic)
    self.update_target(self._target_actor, self._actor)

    if evaluation:
      self._actor = self._actor.eval()
      self._actor.train(False)
      self._critic = self._actor.eval()
      self._critic.train(False)

    self._actor = self._actor.to(self._device)
    self._target_actor = self._target_actor.to(self._device)
    self._critic = self._critic.to(self._device)
    self._target_critic = self._target_critic.to(self._device)

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
    Q_current = self._critic(states, actions)
    # Compute next Q value based on which action target actor would choose
    # Detach variable from the current graph since we don't want gradients for next Q to propagated
    with torch.no_grad():
      target_actions = self._target_actor(states)
      next_max_q = self._target_critic(next_states, target_actions).max(1)[0]

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
    if len(self._memory) < self._batch_size:
      return

    batch = self._memory.sample_transitions(self._batch_size)
    td_error, state_batch_var = self.evaluate(batch)
    loss = self._optimise_wrt_ac(td_error, state_batch_var)

    self.update_target(self._target_critic, self._critic)
    self.update_target(self._target_actor, self._actor)

    return td_error, loss

  def optimise_critic_wrt(self, error):
    self._critic_optimiser.zero_grad()
    error.backward()
    self._critic_optimiser.step()  # Optimize the critic

  def optimise_actor_wrt(self, loss):
    self._actor_optimiser.zero_grad()
    loss.backward()
    self._actor_optimiser.step()  # Optimize the actor

  def update_target(self, target_model, model):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
      target_param.data.copy_(self._target_update_tau
                              * param.data
                              + (1 - self._target_update_tau)
                              * target_param.data
                              )

  def rollout(self,
              initial_state,
              environment,
              render=False,
              train=True,
              **kwargs):
    self._rollout_i += 1

    state = initial_state
    episode_signal = 0
    episode_length = 0

    T = tqdm(count(1), f'Rollout #{self._rollout_i}', leave=False)
    for t in T:
      self._step_i += 1

      action = self.sample_action(state)

      noise = self._random_process.sample()

      action += noise  # Add action space noise for exploration, alternative is parameter space noise

      if self._action_clipping:
        action = np.clip(action, -1.0, 1.0)


      if hasattr(environment,'step'):
        successor_state, signal, terminated, info = environment.step(action)
      else:
        info = environment.react(action)
        successor_state, signal, terminated = info.observables, info.signal, info.terminated

      if render:
        environment.render()

      if self._action_clipping:
        signal = np.clip(action, -1.0, 1.0)

      # successor_state = None
      # if not terminated:  # If environment terminated then there is no successor state

      self._memory.add_transition(          state,
                                            action,
                                            signal,
                                            successor_state,
                                            not terminated
          )
      state = successor_state

      self.update()
      episode_signal += signal

      if terminated:
        episode_length = t
        break

    return episode_signal, episode_length

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
    loss = -self._critic(state_batch, self._actor(state_batch)).mean()
    # loss = -torch.sum(self.critic(state_batch, self.actor(state_batch)))

    self.optimise_actor_wrt(loss)

    # self._memory.batch_update(indices, errors.tolist())  # Cuda trouble

    return loss


  def _build(self, **kwargs) -> None:

    self._actor_arch_parameters['input_size'] = self._input_size
    self._actor_arch_parameters['output_size'] = self._output_size
    self._critic_arch_parameters['input_size'] = self._input_size
    self._critic_arch_parameters['output_size'] = self._output_size

    # Construct actor and critic
    actor = self._actor_arch(**self._actor_arch_parameters).to(self._device)
    target_actor = self._actor_arch(**self._actor_arch_parameters).to(
        self._device
        ).eval()
    critic = self._critic_arch(**self._critic_arch_parameters).to(self._device)
    target_critic = self._critic_arch(**self._critic_arch_parameters).to(
        self._device
        ).eval()

    # Construct the optimizers for actor and critic
    actor_optimizer = self._actor_optimiser_spec.constructor(
        actor.parameters(), **self._critic_optimiser_spec.kwargs
        )
    critic_optimizer = self._critic_optimiser_spec.constructor(
        critic.parameters(), **self._critic_optimiser_spec.kwargs
        )

    self._actor, self._target_actor, self._critic, self._target_critic, self._actor_optimiser, \
    self._critic_optimiser = actor, target_actor, critic, target_critic, actor_optimizer, critic_optimizer

  def _sample_model(self, state, **kwargs):
    state = U.to_tensor([state], device=self._device, dtype=self._state_type)
    with torch.no_grad():
      action = self._actor(state)
    a = action.to('cpu').numpy()
    return a[0]

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
        state =state.observables
      self._random_process.reset()

      if episode_i % stat_frequency == 0:
        draugr.styled_terminal_plot_stats_shared_x(stats, printer=E.write)
        E.set_description(            f'Episode: {episode_i}, '
                                      f'Last signal: {stats.signal[-1]}'
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
    (actor_model, critic_model), stats = agent.train(
        env, config.ROLLOUTS, render=config.RENDER_ENVIRONMENT
        )
  finally:
    listener.stop()

  U.save_model(actor_model, config, name='actor')
  U.save_model(critic_model, config, name='critic')


if __name__ == '__main__':
  import configs.agent_test_configs.test_ddpg_config as C

  test_agent_main(DDPGAgent, C)
