#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from itertools import count
from pathlib import Path
from typing import Union

import numpy

from draugr.writers import MockWriter, TensorBoardPytorchWriter, Writer
from neodroid.environments.environment import Environment
from neodroid.utilities import EnvironmentSnapshot
from neodroidagent.agents import Agent
from neodroidagent.common.memory.experience import Transition
from neodroidagent.common.session_factory.vertical.procedures.procedure_specification import (
    Procedure,
)
from neodroidagent.utilities import is_positive_and_mod_zero
from warg import drop_unused_kws, passes_kws_to

__author__ = "Christian Heider Nielsen"
__all__ = ["rollout_off_policy", "OffPolicyEpisodic"]
__doc__ = "Collects agent experience for episodic off policy training"

from tqdm import tqdm


@drop_unused_kws
def rollout_off_policy(
    agent: Agent,
    initial_state: EnvironmentSnapshot,
    env: Environment,
    *,
    render=False,
    metric_writer: Writer = MockWriter(),
    train_agent=True,
    disallow_random_sample=False,
):
    state = agent.extract_features(initial_state)
    episode_signal = []
    episode_length = []

    T = count(0)
    T = tqdm(T, desc="Step #", leave=False, disable=not render)

    for t in T:
        sample = agent.sample(
            state, deterministic=disallow_random_sample, metric_writer=metric_writer
        )
        act = agent.extract_action(sample)
        snapshot = env.react(act)

        successor_state = agent.extract_features(snapshot)
        signal = agent.extract_signal(snapshot)
        terminated = snapshot.terminated

        if render:
            env.render()

        if train_agent:
            agent.remember(
                signal=signal,
                terminated=terminated,
                sample=sample,
                transition=Transition(state, act, successor_state),
            )

        episode_signal.append(signal)

        if numpy.array(terminated).all():
            episode_length = t
            break

        state = successor_state

    if train_agent:
        agent.update(metric_writer=metric_writer)
    else:
        print("no update")

    ep = numpy.array(episode_signal).sum(axis=0).mean()
    el = episode_length

    if metric_writer:
        metric_writer.scalar("duration", el)
        metric_writer.scalar("signal", ep)

    return ep, el


class OffPolicyEpisodic(Procedure):
    @passes_kws_to(rollout_off_policy)
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
        """
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

                ret, *_ = rollout_off_policy(
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
