#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from itertools import count
from pathlib import Path
from typing import Union

import numpy

from draugr.metrics.accumulation import mean_accumulator, total_accumulator
from draugr.writers import MockWriter, TensorBoardPytorchWriter, Writer
from neodroid.environments.environment import Environment
from neodroid.utilities import EnvironmentSnapshot
from neodroidagent.agents import Agent
from neodroidagent.common.memory.transitions import Transition, TransitionPoint
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
    use_episodic_buffer=True,
):
    state = agent.extract_features(initial_state)
    episode_length = 0

    running_mean_action = mean_accumulator()
    episode_signal = total_accumulator()

    if use_episodic_buffer:
        episode_buffer = []

    for step_i in tqdm(count(), desc="Step #", leave=False, disable=not render):
        sample = agent.sample(
            state, deterministic=disallow_random_sample, metric_writer=metric_writer
        )
        action = agent.extract_action(sample)

        snapshot = env.react(action)

        successor_state = agent.extract_features(snapshot)
        signal = agent.extract_signal(snapshot)
        terminated = snapshot.terminated

        if render:
            env.render()

        if train_agent:
            if use_episodic_buffer:
                a = [
                    TransitionPoint(*s)
                    for s in zip(state, action, successor_state, signal, terminated)
                ]
                episode_buffer.extend(a)
            else:
                agent.remember(
                    signal=signal,
                    terminated=terminated,
                    transition=Transition(state, action, successor_state),
                )

        running_mean_action.send(action.mean())
        episode_signal.send(signal.mean())

        if numpy.array(terminated).all():
            break

        state = successor_state

    if use_episodic_buffer:
        t = TransitionPoint(*zip(*episode_buffer))
        agent.remember(
            signal=t.signal,
            terminated=t.terminal,
            transition=Transition(t.state, t.action, t.successor_state),
        )

    if step_i > 0:
        if train_agent:
            agent.update(metric_writer=metric_writer)
        else:
            print("no update")

        esig = next(episode_signal)
        rma = next(running_mean_action)

        if metric_writer:
            metric_writer.scalar("duration", episode_length)
            metric_writer.scalar("running_mean_action", rma)
            metric_writer.scalar("signal", esig)

        return esig, step_i
    else:
        return 0, 0


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
