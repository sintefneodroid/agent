#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from abc import abstractmethod
from itertools import count
from typing import Any

import numpy
import torch
from numpy import mean
from tqdm import tqdm

from draugr.visualisation import sprint
from draugr.writers import TensorBoardPytorchWriter, MockWriter
from draugr.writers.writer import Writer
from neodroid.environments.environment import Environment
from neodroid.interfaces.specifications import EnvironmentSnapshot
from neodroidagent.architectures import MLP
from neodroidagent.architectures.experimental.merged import MergedInputMLP
from neodroidagent.exceptions.exceptions import ActionSpaceNotSupported
from neodroidagent.interfaces.torch_agent import TorchAgent
from neodroidagent.memory import TransitionBuffer
from neodroidagent.utilities.exploration import OrnsteinUhlenbeckProcess
from warg.gdkc import GDKC

__author__ = 'cnheider'


class ActorCriticAgent(TorchAgent):
  '''
All value iteration agents should inherit from this class
'''

  # region Private

  def __init__(self, *args, **kwargs):
    self._target_update_tau = 3e-3
    self._signal_clipping = False
    self._action_clipping = False

    self._memory_buffer = TransitionBuffer()

    self._actor_optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam,
                                            lr=3e-4,
                                            weight_decay=3e-3,
                                            eps=3e-2
                                            )

    self._critic_optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam,
                                             lr=3e-4,
                                             weight_decay=3e-3,
                                             eps=3e-2
                                             )

    self._actor_arch_spec = GDKC(MLP,
                                 input_shape=None,  # Obtain from environment
                                 hidden_layers=None,
                                 output_shape=None,  # Obtain from environment
                                 )

    self._critic_arch_spec = GDKC(MergedInputMLP,
                                  input_shape=None,  # Obtain from environment
                                  hidden_layers=None,
                                  output_shape=1,  # Obtain from environment
                                  )

    self._random_process_spec = GDKC(constructor=OrnsteinUhlenbeckProcess,
                                     theta=0.15,
                                     sigma=0.2
                                     )

    super().__init__(*args, **kwargs)

  def __build__(self, env: Environment,
                metric_writer: Writer = MockWriter(),
                print_model_repr=True,
                **kwargs) -> None:
    # Construct actor and critic
    self._actor = self._actor_arch_spec().to(self._device)
    self._target_actor = self._actor_arch_spec().to(self._device).eval()

    self._critic = self._critic_arch_spec().to(self._device)
    self._target_critic = self._critic_arch_spec().to(self._device).eval()

    self._random_process = self._random_process_spec(
        sigma=mean([r.span for r in env.action_space.ranges])
        )

    # Construct the optimizers for actor and critic
    self._actor_optimiser = self._actor_optimiser_spec(self._actor.parameters())
    self._critic_optimiser = self._critic_optimiser_spec(self._critic.parameters())

    if metric_writer:
      dummy_in = torch.rand(1, *self.input_shape)

      model = copy.deepcopy(self._critic)
      model.to('cpu')
      if isinstance(metric_writer, TensorBoardPytorchWriter):
        metric_writer.graph(model, dummy_in)

      model = copy.deepcopy(self._actor)
      model.to('cpu')
      if isinstance(metric_writer, TensorBoardPytorchWriter):
        metric_writer.graph(model, dummy_in)

    if print_model_repr:
      sprint(f'Critic: {self._critic}',
             highlight=True,
             color='cyan')
      sprint(f'Actor: {self._actor}',
             highlight=True,
             color='cyan')

  def _post_io_inference(self, env: Environment):

    self._actor_arch_spec.kwargs['input_shape'] = self._input_shape
    if env.action_space.is_discrete:
      raise ActionSpaceNotSupported()

    self._actor_arch_spec.kwargs['output_shape'] = self._output_shape

    self._critic_arch_spec.kwargs['input_shape'] = (*self._input_shape, *self._output_shape)
    # self._actor_arch_spec = GDCS(MergedInputMLP, self._critic_arch_spec.kwargs)
    self._critic_arch_spec.kwargs['output_shape'] = 1

  # endregion

  # region Public

  @property
  def models(self):
    return {'_actor':self._actor, '_critic':self._critic}

  def rollout(self,
              initial_state: EnvironmentSnapshot,
              environment: Environment,
              *,
              metric_writer: Writer = MockWriter(),
              render: bool = False,
              train: bool = True,
              **kwargs):

    state = initial_state.observables
    episode_signal = []
    episode_length = 0

    T = tqdm(count(1), f'Rollout #{self._update_i}', leave=False, disable=not render)
    for t in T:
      self._sample_i += 1

      action = self.sample(state, disallow_random_sample=not train)
      snapshot = environment.react(action)

      successor_state, signal, terminated = snapshot.observables, snapshot.signal, snapshot.terminated

      if render:
        environment.render()

      # successor_state = None
      # if not terminated:  # If environment terminated then there is no successor state

      if self._signal_clipping:
        signal = numpy.clip(signal, -1.0, 1.0)

      if train:
        self._memory_buffer.add_transition(state,
                                           action,
                                           signal,
                                           successor_state,
                                           terminated
                                           )
      state = successor_state

      if train:
        self.update()
      episode_signal.append(signal)

      if numpy.array(terminated).all():
        episode_length = t
        break

    es = numpy.array(episode_signal).mean()
    el = episode_length

    if metric_writer:
      metric_writer.scalar('duration', el, self._update_i)
      metric_writer.scalar('signal', es, self._update_i)

    return es, el

  # endregion

  # region Protected

  def _sample(self, state, **kwargs):
    return self._sample_model(state, **kwargs)

  @abstractmethod
  def _sample_model(self, state, **kwargs) -> Any:
    raise NotImplementedError

  # endregion
