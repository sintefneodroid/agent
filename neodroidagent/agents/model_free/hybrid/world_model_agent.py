#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from neodroidagent.agents.model_free.hybrid.ddpg_agent import DDPGAgent
from neodroidagent.architectures import MLP
from neodroidagent.architectures.experimental.merged import MergedInputMLP
from neodroidagent.interfaces.specifications import ArchitectureSpecification, GDCS
from neodroidagent.memory import TransitionBuffer
from neodroidagent.training.agent_session_entry_point import agent_session_entry_point
from neodroidagent.training.procedures import to_tensor, train_episodically
from neodroidagent.training.sessions.parallel_training import parallelised_training
from neodroidagent.utilities.exploration.sampling import OrnsteinUhlenbeckProcess
from agents import Agent
from draugr.writers.writer import Writer
from warg.named_ordered_dictionary import NOD

__author__ = 'cnheider'

tqdm.monitor_interval = 0


class WorldModelAgent(Agent):
  '''
  As of https://worldmodels.github.io/, https://arxiv.org/abs/1803.10122

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

  def _train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, terminals: np.ndarray):
    pass


# region Test


def wm_test(rollouts=None, skip=True):
  import neodroidagent.configs.agent_test_configs.ddpg_test_config as C
  if rollouts:
    C.ROLLOUTS = rollouts

  agent_session_entry_point(DDPGAgent,
                            C,
                            training_session=parallelised_training(training_procedure=train_episodically,
                                                                   auto_reset_on_terminal_state=True),
                            parse_args=False,
                            skip_confirmation=skip)


def wm_run(rollouts=None, skip=True):
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

  wm_run()
# endregion
