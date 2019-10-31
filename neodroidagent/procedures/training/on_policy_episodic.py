#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count
from pathlib import Path
from typing import Union

import numpy
import torch

from draugr.drawers.drawer import Drawer, MockDrawer
from draugr.writers import TensorBoardPytorchWriter, Writer, MockWriter
from neodroid.environments.environment import Environment
from neodroid.utilities import EnvironmentSnapshot
from neodroidagent.agents.torch_agents.model_free.on_policy.policy_agent import PolicyAgent

__author__ = 'Christian Heider Nielsen'
__doc__ = ''
from tqdm import tqdm

from neodroidagent.utilities.specifications import TR, Procedure


def rollout_on_policy(agent: PolicyAgent,
                      initial_state: EnvironmentSnapshot,
                      environment: Environment,
                      *,
                      render: bool = False,
                      metric_writer: Writer = MockWriter(),
                      rollout_drawer: Drawer = MockDrawer(),
                      train: bool = True,
                      max_length: int = None,
                      disable_stdout: bool = False,
                      **kwargs):
  '''Perform a single rollout until termination in environment

  :param rollout_drawer:
  :param disable_stdout:
  :param metric_writer:
  :type max_length: int
  :param max_length:
  :type train: bool
  :type render: bool
  :param initial_state: The initial state observation in the environment
  :param environment: The environment the agent interacts with
  :param render: Whether to render environment interaction
  :param train: Whether the agent should use the rollout to update its model
  :param kwargs:
  :return:
    -episode_signal (:py:class:`float`) - first output
    -episode_length-
    -average_episode_entropy-
  '''

  episode_signal = []
  episode_length = 0
  episode_entropy = []

  state = initial_state.observables

  '''
  with draugr.scroll_plot_class(self._distribution_regressor.output_shape,
                                render=render,
                                window_length=66) as s:
                                '''
  for t in tqdm(count(1), f'Update #{agent.update_i}', leave=False, disable=disable_stdout):
    action, action_log_prob, entropy = agent.sample(state)

    snapshot = environment.react(action)

    state, signal, terminated = snapshot.observables, snapshot.signal, snapshot.terminated

    if train:
      agent.remember(signal=signal, action_log_prob=action_log_prob, entropy=entropy)

    episode_signal.append(signal)
    episode_entropy.append(entropy.to('cpu').numpy())

    if render:
      environment.render()
      # s.draw(to_one_hot(self._distribution_regressor.output_shape, action)[0])

    if numpy.array(terminated).all() or (max_length and t > max_length):
      episode_length = t
      break

  if train:
    agent.update()

  ep = numpy.array(episode_signal).sum(axis=0).mean()
  el = episode_length
  ee = numpy.array(episode_entropy).mean(axis=0).mean()

  if metric_writer:
    metric_writer.scalar('duration', el, agent.update_i)
    metric_writer.scalar('signal', ep, agent.update_i)
    metric_writer.scalar('entropy', ee, agent.update_i)

  return ep, el, ee


class OnPolicyEpisodic(Procedure):

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

          rollout_on_policy(self.agent,
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
