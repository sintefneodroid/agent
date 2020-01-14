#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count
from pathlib import Path
from typing import Union

import numpy

from draugr.writers import MockWriter, TensorBoardPytorchWriter, Writer
from neodroid.environments.environment import Environment
from neodroid.utilities import EnvironmentSnapshot
from neodroidagent.agents.agent import Agent
from neodroidagent.common.procedures.procedure_specification import Procedure
from neodroidagent.utilities.bool_tests import is_set_mod_zero_ret_alt
from warg.kw_passing import drop_unused_kws, passes_kws_to

__author__ = "Christian Heider Nielsen"
__all__ = ["rollout_off_policy", "OffPolicyEpisodic"]
__doc__ = "Collects agent experience for episodic off policy training"

from tqdm import tqdm


@drop_unused_kws
def rollout_off_policy(
    agent: Agent,
    initial_state: EnvironmentSnapshot,
    environment: Environment,
    *,
    render=False,
    metric_writer: Writer = MockWriter(),
    train_agent=True,
    disallow_random_sample=False,
):
    state = initial_state.observables
    episode_signal = []
    episode_length = []

    T = count(0)
    T = tqdm(T, desc="Step #", leave=False, disable=not render)

    for t in T:
        action = agent.sample(
            state, no_random=disallow_random_sample, metric_writer=metric_writer
        )
        snapshot = environment.react(action)

        next_state, signal, terminated = (
            snapshot.observables,
            snapshot.signal,
            snapshot.terminated,
        )

        if render:
            environment.render()

        if train_agent:
            agent.remember(
                state=state,
                action=action,
                signal=signal,
                next_state=next_state,
                terminated=terminated,
            )

        episode_signal.append(signal)

        if numpy.array(terminated).all():
            episode_length = t
            break

        state = next_state

    if train_agent:
        agent.update()
    else:
        print("no update")

    ep = numpy.array(episode_signal).sum(axis=0).mean()
    el = episode_length

    if metric_writer:
        metric_writer.scalar("duration", el, agent._update_i)
        metric_writer.scalar("signal", ep, agent._update_i)
        metric_writer.scalar(
            "current_eps_threshold", agent._current_eps_threshold, agent._update_i
        )

    return ep, el


class OffPolicyEpisodic(Procedure):
    @drop_unused_kws
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

            for episode_i in E:
                initial_state = self.environment.reset()

                rollout_off_policy(
                    self.agent,
                    initial_state,
                    self.environment,
                    render=is_set_mod_zero_ret_alt(render_frequency, episode_i),
                    metric_writer=is_set_mod_zero_ret_alt(
                        stat_frequency, episode_i, ret=metric_writer
                    ),
                    disable_stdout=disable_stdout,
                    **kwargs,
                )

                if self.early_stop:
                    break
