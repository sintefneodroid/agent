#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union

import torch

from draugr.writers import TensorBoardPytorchWriter
from neodroid.environments import NeodroidEnvironment, VectorEnvironment
from neodroidagent.interfaces.agent import Agent
from neodroidagent.interfaces.torch_agent import TorchAgent

__author__ = 'cnheider'
__doc__ = ''
from tqdm import tqdm

from neodroidagent.interfaces.specifications import TR


def train_episodically(agent: TorchAgent,
                       environment: VectorEnvironment,
                       *,
                       log_directory: Union[str, Path],
                       rollouts: int = 1000,
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
    :type environment: NeodroidEnvironment
    :param rollouts: How many rollouts to train for
    :type rollouts: int
    :param render_frequency: How often to render environment
    :type render_frequency: int
    :param stat_frequency: How often to write statistics
    :type stat_frequency: int
    :return: A training resume containing the trained agents models and some statistics
    :rtype: TR
  '''

  E = range(1, rollouts)
  E = tqdm(E, leave=False)
  # with torchsnooper.snoop():
  with torch.autograd.detect_anomaly():
    with TensorBoardPytorchWriter(str(log_directory)) as metric_writer:
      for episode_i in E:
        initial_state = environment.reset()

        agent.rollout(initial_state,
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

        if agent.end_training:
          break

  return TR(agent.models, None)
