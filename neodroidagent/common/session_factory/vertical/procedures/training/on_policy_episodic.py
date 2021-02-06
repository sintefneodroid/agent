#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import math
from itertools import count
from pathlib import Path
from typing import Union

import numpy
import torch
import torchsnooper
from draugr.drawers import DiscreteScrollPlot

from draugr.drawers.mpldrawer import MplDrawer, MockDrawer
from draugr.metrics.accumulation import mean_accumulator, total_accumulator
from draugr.writers import MockWriter, Writer
from draugr.torch_utilities import TensorBoardPytorchWriter
from neodroid.environments.environment import Environment
from neodroid.utilities import EnvironmentSnapshot, to_one_hot
from neodroidagent.agents.agent import Agent
from neodroidagent.common.session_factory.vertical.procedures.procedure_specification import (
    Procedure,
)
from warg import is_positive_and_mod_zero, drop_unused_kws, passes_kws_to

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
    rollout_ith: int = None,
    render_environment: bool = False,
    metric_writer: Writer = MockWriter(),
    rollout_drawer: MplDrawer = MockDrawer(),
    train_agent: bool = True,
    max_length: int = None,
    disable_stdout: bool = False,
):
    """Perform a single rollout until termination in environment

      :param rollout_ith:
    :param agent:
    :param rollout_drawer:
    :param disable_stdout:
    :param metric_writer:
    :type max_length: int
    :param max_length:
    :type train_agent: bool
    :type render_environment: bool
    :param initial_snapshot: The initial state observation in the environment
    :param env: The environment the agent interacts with
    :param render_environment: Whether to render environment interaction
    :param train_agent: Whether the agent should use the rollout to update its model
    :return:
    -episode_signal (:py:class:`float`) - first output
    -episode_length-
    -average_episode_entropy-"""

    state = agent.extract_features(initial_snapshot)
    running_mean_action = mean_accumulator()
    episode_signal = total_accumulator()

    rollout_description = f"Rollout"
    if rollout_ith:
        rollout_description += f" #{rollout_ith}"
    for step_i in tqdm(
        count(1),
        rollout_description,
        unit="th step",
        leave=False,
        disable=disable_stdout,
        postfix=f"Agent update #{agent.update_i}",
    ):
        sample = agent.sample(state)
        action = agent.extract_action(sample)

        snapshot = env.react(action)

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

        running_mean_action.send(action.mean())
        episode_signal.send(signal.mean())

        if render_environment:
            env.render()
            if rollout_drawer:
                if env.action_space.is_discrete:
                    action = to_one_hot(agent.output_shape, action)
                rollout_drawer.draw(action)

        if numpy.array(terminated).all() or (max_length and step_i > max_length):
            break

    if train_agent:
        agent.update(metric_writer=metric_writer)
    else:
        logging.info("no update")

    episode_return = next(episode_signal)
    rma = next(running_mean_action)

    if metric_writer:
        metric_writer.scalar("duration", step_i, agent.update_i)
        metric_writer.scalar("running_mean_action", rma, agent.update_i)
        metric_writer.scalar("signal", episode_return, agent.update_i)

    return episode_return, step_i


class OnPolicyEpisodic(Procedure):
    @passes_kws_to(rollout_on_policy)
    def __call__(
        self,
        *,
        iterations: int = 1000,
        render_frequency: int = 100,
        stat_frequency: int = 10,
        disable_stdout: bool = False,
        metric_writer: Writer = MockWriter(),
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
        :rtype: TR"""

        E = range(1, iterations)
        E = tqdm(E, desc="Rollout #", leave=False)

        best_episode_return = -math.inf
        for episode_i in E:
            initial_state = self.environment.reset()

            kwargs.update(
                render_environment=is_positive_and_mod_zero(render_frequency, episode_i)
            )
            ret, *_ = rollout_on_policy(
                self.agent,
                initial_state,
                self.environment,
                rollout_íth=episode_i,
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
