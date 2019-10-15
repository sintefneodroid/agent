#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy
from neodroidagent.training.procedures import Agent, episodic_training
from tqdm import tqdm

from neodroidagent.agents.torch_agents.model_free import DDPGAgent
from neodroidagent.sessions.session_entry_point import session_entry_point
from neodroidagent.sessions.single_agent.parallel import ParallelSession

__author__ = 'Christian Heider Nielsen'

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

  def _train(self, states: numpy.ndarray, actions: numpy.ndarray, rewards: numpy.ndarray,
             terminals: numpy.ndarray):
    pass


# region Test


def wm_test(rollouts=None, skip=True):
  import neodroidagent.configs.agent_test_configs.ddpg_test_config as C
  if rollouts:
    C.ROLLOUTS = rollouts

  session_entry_point(DDPGAgent,
                      C,
                      session=ParallelSession(procedure=episodic_training,
                                              auto_reset_on_terminal_state=True),
                      parse_args=False,
                      skip_confirmation=skip)


def wm_run(rollouts=None, skip=True):
  import neodroidagent.configs.agent_test_configs.ddpg_test_config as C
  if rollouts:
    C.ROLLOUTS = rollouts

  session_entry_point(DDPGAgent,
                      C,
                      session=ParallelSession(procedure=episodic_training,
                                              auto_reset_on_terminal_state=True),
                      parse_args=False,
                      skip_confirmation=skip)


if __name__ == '__main__':

  wm_run()
# endregion
