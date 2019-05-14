#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod
from itertools import count
from typing import Any

import draugr
from tqdm import tqdm

from agent.utilities.specifications.generalised_delayed_construction_specification import GDCS

__author__ = 'cnheider'
import torch

from agent import utilities as U
from agent.agents.abstract.torch_agent import TorchAgent
import numpy as np


class ActorCriticAgent(TorchAgent):
  '''
All value iteration agents should inherit from this class
'''

  # region Private

  def __init__(self, *args, **kwargs):
    self._actor_arch_spec: GDCS = None
    self._critic_arch_spec: GDCS = None

    self._target_update_tau = 3e-3
    self._signal_clipping = False
    self._action_clipping = False

    self._memory_buffer = U.TransitionBuffer()

    self._actor_optimiser_spec: GDCS = GDCS(constructor=torch.optim.Adam,
                                            kwargs={'lr':3e-4}
                                            )

    self._critic_optimiser_spec: GDCS = GDCS(constructor=torch.optim.Adam,
                                             kwargs={'lr':          3e-3,
                                                     'weight_decay':3e-2
                                                     }
                                             )

    super().__init__(*args, **kwargs)

  def _build(self, **kwargs) -> None:
    # Construct actor and critic
    self._actor = self._actor_arch_spec.constructor(**self._actor_arch_spec.kwargs).to(self._device)
    self._target_actor = self._actor_arch_spec.constructor(**self._actor_arch_spec.kwargs).to(
        self._device).eval()

    self._critic = self._critic_arch_spec.constructor(**self._critic_arch_spec.kwargs).to(self._device)
    self._target_critic = self._critic_arch_spec.constructor(**self._critic_arch_spec.kwargs).to(
        self._device).eval()

    # Construct the optimizers for actor and critic
    self._actor_optimiser = self._actor_optimiser_spec.constructor(self._actor.parameters(),
                                                                   **self._actor_optimiser_spec.kwargs
                                                                   )
    self._critic_optimiser = self._critic_optimiser_spec.constructor(self._critic.parameters(),
                                                                     **self._critic_optimiser_spec.kwargs
                                                                     )

    actor_num_params = sum(param.numel() for param in self._actor.parameters())
    critic_num_params = sum(param.numel() for param in self._critic.parameters())

    actor_num_trainable_params = sum(
        p.numel() for p in self._actor.parameters() if p.requires_grad)

    critic_num_trainable_params = sum(
        p.numel() for p in self._critic.parameters() if p.requires_grad)

    draugr.sprint(f'trainable/actor_num_params: {actor_num_trainable_params}/{actor_num_params}\n',
                  highlight=True, color='cyan')
    draugr.sprint(f'trainable/critic_num_params: {critic_num_trainable_params}/{critic_num_params}\n',
                  highlight=True, color='magenta')

  def _maybe_infer_sizes(self, env):
    super()._maybe_infer_sizes(env)

    if ('input_size' not in self._actor_arch_spec.kwargs or
        not self._actor_arch_spec.kwargs['input_size']):
      self._actor_arch_spec.kwargs['input_size'] = self._input_size

    if ('hidden_layers' not in self._actor_arch_spec.kwargs or
        not self._actor_arch_spec.kwargs['hidden_layers']):
      self._actor_arch_spec.kwargs['hidden_layers'] = self._hidden_layers

    if ('output_size' not in self._actor_arch_spec.kwargs or
        not self._actor_arch_spec.kwargs['output_size']):
      self._actor_arch_spec.kwargs['output_size'] = self._output_size

    if ('input_size' not in self._critic_arch_spec.kwargs or
        not self._critic_arch_spec.kwargs['input_size']):
      self._critic_arch_spec.kwargs['input_size'] = self._input_size

    if ('hidden_layers' not in self._critic_arch_spec.kwargs or
        not self._critic_arch_spec.kwargs['hidden_layers']):
      self._critic_arch_spec.kwargs['hidden_layers'] = self._hidden_layers

    if ('output_size' not in self._critic_arch_spec.kwargs or
        not self._critic_arch_spec.kwargs['output_size']):
      self._critic_arch_spec.kwargs['output_size'] = self._output_size

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

    self._build(**kwargs)

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
    episode_signal = []
    episode_length = 0

    T = tqdm(count(1), f'Rollout #{self._rollout_i}', leave=False)
    for t in T:
      self._step_i += 1

      action = self.sample_action(state)

      successor_state, signal, terminated, *_ = environment.react(action)

      if render:
        environment.render()

      # successor_state = None
      # if not terminated:  # If environment terminated then there is no successor state

      if self._signal_clipping:
        signal = np.clip(signal, -1.0, 1.0)

      if train:
        self._memory_buffer.add_transition(state,
                                           action,
                                           signal,
                                           successor_state,
                                           [not t for t in terminated]
                                           )
      state = successor_state

      if train:
        self.update()
      episode_signal.append(signal)

      if terminated.all():
        episode_length = t
        break

    es = np.array(episode_signal).mean()
    el = episode_length
    return es, episode_length

  # endregion
