#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count
from pathlib import Path
from typing import Union, Any

import numpy
import torch

from draugr import copy_state
from draugr.drawers.drawer import Drawer, MockDrawer
from draugr.writers import TensorBoardPytorchWriter, Writer, MockWriter
from neodroid.environments.environment import Environment
from neodroid.utilities import EnvironmentSnapshot
from neodroidagent.agents.torch_agents.model_free.off_policy.value_agent import ValueAgent
from neodroidagent.agents.torch_agents.model_free.on_policy.policy_agent import PolicyAgent

__author__ = 'Christian Heider Nielsen'
__doc__ = ''
from tqdm import tqdm

from neodroidagent.utilities.specifications import TR, Procedure, VectorUnityEnvironment


def rollout_off_policy(agent: ValueAgent,
                       initial_state: EnvironmentSnapshot,
                       environment: Environment,
                       *,
                       render=False,
                       metric_writer: Writer = MockWriter(),
                       train=True,
                       disallow_random_sample=False,
                       learning_frequency=0,
                       initial_observation_period=0,  # 10e-2,
                       **kwargs):
  state = initial_state.observables
  episode_signal = []
  episode_length = []

  T = count(1)
  T = tqdm(T, f'Rollout #{agent.update_i}', leave=False, disable=not render)

  for t in T:
    action = agent.sample(state, no_random=disallow_random_sample, metric_writer=metric_writer)
    snapshot = environment.react(action)

    next_state, signal, terminated = snapshot.observables, snapshot.signal, snapshot.terminated

    if render:
      environment.render()

    if train:
      agent.remember(state=state,
                     action=action,
                     signal=signal,
                     next_state=next_state,
                     terminated=terminated)

      if (len(agent._memory_buffer) >= agent._batch_size
        and agent.update_i > initial_observation_period
        and (learning_frequency == 0 or agent.update_i % learning_frequency == 0)
      ):
        agent.update()

    episode_signal.append(signal)

    if numpy.array(terminated).all():
      episode_length = t
      break

    state = next_state

  ep = numpy.array(episode_signal).sum(axis=0).mean()
  el = episode_length

  if metric_writer:
    metric_writer.scalar('duration', el, agent._update_i)
    metric_writer.scalar('signal', ep, agent._update_i)
    metric_writer.scalar('current_eps_threshold', agent._current_eps_threshold, agent._update_i)

  return ep, el


class OffPolicyEpisodic(Procedure):

  def __call__(self,
               *,
               log_directory: Union[str, Path],
               iterations: int = 1000,
               render_frequency: int = 100,
               stat_frequency: int = 10,
               disable_stdout: bool = False,
               **kwargs
               ) -> TR:
    '''
      :param log_directory:
      :param agent: The learning agent
      :type agent: Agent
      :param disable_stdout: Whether to disable stdout statements or not
      :type disable_stdout: bool
      :param environment: The environment the agent should interact with
      :type environment: UnityEnvironment
      :param iterations: How many iterations to train for
      :type iterations: int
      :param render_frequency: How often to render environment
      :type render_frequency: int
      :param stat_frequency: How often to write statistics
      :type stat_frequency: int
      :return: A training resume containing the trained agents models and some statistics
      :rtype: TR
    '''

    # with torchsnooper.snoop():
    with torch.autograd.detect_anomaly():
      with TensorBoardPytorchWriter(log_directory) as metric_writer:
        E = range(1, iterations)
        E = tqdm(E, leave=False)

        for episode_i in E:
          initial_state = self.environment.reset()

          rollout_off_policy(self.agent,
                             initial_state,
                             self.
                             environment,
                             render=(True
                                     if (render_frequency and
                                         episode_i % render_frequency == 0)
                                     else
                                     False),
                             metric_writer=(metric_writer
                                            if (stat_frequency and
                                                episode_i % stat_frequency == 0)
                                            else
                                            None),
                             disable_stdout=disable_stdout)

          if self.early_stop:
            break
