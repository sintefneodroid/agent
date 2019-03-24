#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod
from itertools import count
from typing import Any

from tqdm import tqdm

__author__ = 'cnheider'
import torch

import utilities as U
from agents.abstract.torch_agent import TorchAgent
import numpy as np


class ActorCriticAgent(TorchAgent):
  '''
All value iteration agents should inherit from this class
'''

  # region Private

  def __init__(self, *args, **kwargs):
    self._actor_arch = None
    self._critic_arch = None
    self._actor_arch_parameters = None
    self._critic_arch_parameters = None

    self._target_update_tau = 3e-3
    self._signal_clipping = False
    self._action_clipping = False

    self._memory_buffer = U.ExpandableCircularBuffer()

    self._optimiser_type = torch.optim.Adam

    self._actor_optimiser_spec = U.OptimiserSpecification(constructor=self._optimiser_type,
                                                          kwargs={'lr':3e-4}
                                                          )
    self._critic_optimiser_spec = U.OptimiserSpecification(constructor=self._optimiser_type,
                                                           kwargs={'lr':3e-3,
                                                                       'weight_decay':3e-2}
                                                           )

    super().__init__(*args, **kwargs)

  def __build__(self, **kwargs) -> None:
    # Construct actor and critic
    self._actor = self._actor_arch(**self._actor_arch_parameters).to(self._device)
    self._target_actor = self._actor_arch(**self._actor_arch_parameters).to(self._device).eval()

    self._critic = self._critic_arch(**self._critic_arch_parameters).to(self._device)
    self._target_critic = self._critic_arch(**self._critic_arch_parameters).to(self._device).eval()

    # Construct the optimizers for actor and critic
    self._actor_optimiser = self._actor_optimiser_spec.constructor(self._actor.parameters(),
                                                                   **self._actor_optimiser_spec.kwargs
                                                                   )
    self._critic_optimiser = self._critic_optimiser_spec.constructor(self._critic.parameters(),
                                                                     **self._critic_optimiser_spec.kwargs
                                                                     )

  def __maybe_infer_sizes(self, env):
    super().__maybe_infer_sizes(env)

    if ('input_size' not in self._actor_arch_parameters or
        not self._actor_arch_parameters['input_size']):
      self._actor_arch_parameters['input_size'] = self._observation_size

    if ('hidden_layers' not in self._actor_arch_parameters or
        not self._actor_arch_parameters['hidden_layers']):
      self._actor_arch_parameters['hidden_layers'] = self._hidden_layers

    if ('output_size' not in self._actor_arch_parameters or
        not self._actor_arch_parameters['output_size']):
      self._actor_arch_parameters['output_size'] = self._action_size

    if ('input_size' not in self._critic_arch_parameters or
        not self._critic_arch_parameters['input_size']):
      self._critic_arch_parameters['input_size'] = self._observation_size

    if ('hidden_layers' not in self._critic_arch_parameters or
        not self._critic_arch_parameters['hidden_layers']):
      self._critic_arch_parameters['hidden_layers'] = self._hidden_layers

    if ('output_size' not in self._critic_arch_parameters or
        not self._critic_arch_parameters['output_size']):
      self._critic_arch_parameters['output_size'] = self._action_size

  # endregion

  # region Public

  def save(self, C):
    U.save_model(self._actor, C, name='actor')
    U.save_model(self._critic, C, name='policy')

  def load(self,
           model_path,
           evaluation=False,
           **kwargs):
    print('loading latest model: ' + model_path)

    self.__build__(**kwargs)

    self._actor.load_state_dict(torch.load(f'actor-{model_path}'))
    self._critic.load_state_dict(torch.load(f'critic-{model_path}'))

    self.update_target(target_model=self._target_critic,
                       source_model=self._critic,
                       target_update_tau=self._target_update_tau)
    self.update_target(target_model=self._target_actor,
                       source_model=self._actor,
                       target_update_tau=self._target_update_tau)

    if evaluation:
      self._actor = self._actor.eval()
      self._actor.train(False)
      self._critic = self._actor.eval()
      self._critic.train(False)

    self._actor = self._actor.to(self._device)
    self._target_actor = self._target_actor.to(self._device)
    self._critic = self._critic.to(self._device)
    self._target_critic = self._target_critic.to(self._device)

  @staticmethod
  def update_target(*, target_model, source_model, target_update_tau=3e-3):
    assert 0.0 <= target_update_tau <= 1.0
    for target_param, param in zip(target_model.parameters(), source_model.parameters()):
      target_param.data.copy_(target_update_tau
                              * param.data
                              + (1 - target_update_tau)
                              * target_param.data
                              )

  # endregion

  # region Protected

  def sample_action(self, state, *args, **kwargs):
    return self._sample_model(state, *args, *kwargs)

  @abstractmethod
  def _sample_model(self, state, *args, **kwargs) -> Any:
    raise NotImplementedError

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

      if hasattr(environment, 'step'):
        successor_state, signal, terminated, info = environment.step(action)
      else:
        info = environment.react(action)
        successor_state, signal, terminated = info.observables, info.signal, info.terminated

      if render:
        environment.render()

      # successor_state = None
      # if not terminated:  # If environment terminated then there is no successor state

      if self._signal_clipping:
        signal = np.clip(signal, -1.0, 1.0)

      self._memory_buffer.add_transition(state,
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

  # endregion
