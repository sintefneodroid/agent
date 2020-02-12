#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from itertools import count
from pathlib import Path
from typing import Union

import numpy

from neodroidagent.common import SampleTrajectoryPoint
from neodroidagent.common.session_factory.vertical.procedures.procedure_specification import (
    Procedure,
)
from draugr.drawers.drawer import Drawer, MockDrawer
from draugr.writers import TensorBoardPytorchWriter, MockWriter, Writer
from neodroid.environments.environment import Environment
from neodroid.utilities import EnvironmentSnapshot, to_one_hot
from neodroidagent.agents.agent import Agent
from neodroidagent.utilities.misc import is_positive_and_mod_zero

from warg.kw_passing import drop_unused_kws, passes_kws_to

__author__ = "Christian Heider Nielsen"
__all__ = ["rollout_on_policy", "OnPolicyEpisodic"]
__doc__ = "Collects agent experience for episodic on policy training"

from tqdm import tqdm


@drop_unused_kws
def rollout_on_policy(
    agent: Agent,
    initial_snapshot: EnvironmentSnapshot,
    env: Environment,
    *,
    render: bool = False,
    metric_writer: Writer = MockWriter(),
    rollout_drawer: Drawer = MockDrawer(),
    train_agent: bool = True,
    max_length: int = None,
    disable_stdout: bool = False,
):
    """Perform a single rollout until termination in environment

  :param agent:
  :param rollout_drawer:
  :param disable_stdout:
  :param metric_writer:
  :type max_length: int
  :param max_length:
  :type train_agent: bool
  :type render: bool
  :param initial_snapshot: The initial state observation in the environment
  :param env: The environment the agent interacts with
  :param render: Whether to render environment interaction
  :param train_agent: Whether the agent should use the rollout to update its model
  :return:
    -episode_signal (:py:class:`float`) - first output
    -episode_length-
    -average_episode_entropy-
  """

    episode_signal = []
    episode_length = 0

    state = agent.extract_features(initial_snapshot)

    for t in tqdm(
        count(1), f"Update #{agent.update_i}", leave=False, disable=disable_stdout
    ):
        sample = agent.sample(state)
        snapshot = env.react(agent.extract_action(sample))

        successor_state = agent.extract_features(snapshot)
        terminated = snapshot.terminated
        signal = agent.extract_signal(snapshot)

        if train_agent:
            agent.remember(
                state=state,
                signal=signal,
                terminated=terminated,
                sample=sample,
                successor_state=successor_state,
            )

        state = successor_state

        episode_signal.append(signal)

        if render:
            env.render()
            # if env.action_space.is_discrete and rollout_drawer:
            #  rollout_drawer.draw(to_one_hot(agent.output_shape, action)[0])

        if numpy.array(terminated).all() or (max_length and t > max_length):
            episode_length = t
            break

    if train_agent:
        agent.update(metric_writer=metric_writer)
    else:
        print("no update")

    ep = numpy.array(episode_signal).sum(axis=0).mean()
    el = episode_length

    if metric_writer:
        metric_writer.scalar("duration", el, agent.update_i)
        metric_writer.scalar("signal", ep, agent.update_i)

    return ep, el


class OnPolicyEpisodic(Procedure):
    @passes_kws_to(rollout_on_policy)
    def __call__(
        self,
        *,
        log_directory: Union[str, Path],
        iterations: int = 1000,
        render_frequency: int = 100,
        stat_frequency: int = 10,
        disable_stdout: bool = False,
        **kwargs,
    ):
        r"""
    :param log_directory:
    :param disable_stdout: Whether to disable stdout statements or not
    :type disable_stdout: bool
    :param iterations: How many iterations to train for
    :type iterations: int
    :param render_frequency: How often to render environment
    :type render_frequency: int
    :param stat_frequency: How often to write statistics
    :type stat_frequency: int
    :return: A training resume containing the trained agents models and some statistics
    :rtype: TR
    """

        # with torchsnooper.snoop():
        # with torch.autograd.detect_anomaly():
        with TensorBoardPytorchWriter(log_directory) as metric_writer:
            E = range(1, iterations)
            E = tqdm(E, desc="Rollout #", leave=False)

            best_episode_return = -math.inf
            for episode_i in E:
                initial_state = self.environment.reset()

                ret, *_ = rollout_on_policy(
                    self.agent,
                    initial_state,
                    self.environment,
                    render=is_positive_and_mod_zero(render_frequency, episode_i),
                    metric_writer=is_positive_and_mod_zero(
                        stat_frequency, episode_i, ret=metric_writer
                    ),
                    disable_stdout=disable_stdout,
                    **kwargs,
                )

                if best_episode_return < ret:
                    best_episode_return = ret
                    self.call_on_improvement_callbacks(**kwargs)

                if self.early_stop:
                    break
