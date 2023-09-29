#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__all__ = ["rollout_on_policy", "OnPolicyEpisodic"]
__doc__ = "Collects agent experience for episodic on policy training"

import logging
import math
import random
import time
from itertools import count
from typing import Optional

import numpy
from draugr.drawers import MockDrawer, MplDrawer
from draugr.metrics import mean_accumulator, total_accumulator
from draugr.opencv_utilities import blit_fps, blit_numbering_raster_sequence
from draugr.visualisation import progress_bar
from draugr.writers import MockWriter, VideoInputDimsEnum, VideoWriterMixin, Writer
from neodroid.environments.environment import Environment
from neodroid.utilities import EnvironmentSnapshot, to_one_hot
from neodroidagent.agents.agent import Agent
from neodroidagent.common.session_factory.vertical.procedures.procedure_specification import (
    Procedure,
)
from neodroidagent.utilities.misc.common_metrics import CommonEnvironmentScalarEnum
from trolls.render_mode import RenderModeEnum
from warg import drop_unused_kws, is_positive_and_mod_zero, passes_kws_to


@drop_unused_kws
def rollout_on_policy(
    agent: Agent,
    initial_snapshot: EnvironmentSnapshot,
    env: Environment,
    *,
    rollout_ith: int = None,
    render_mode: RenderModeEnum = RenderModeEnum.none,
    metric_writer: Optional[Writer] = MockWriter(),
    drawer: MplDrawer = MockDrawer(),
    train_agent: bool = True,
    max_length: int = None,
    disable_stdout: bool = False,
    select_random_single_render: bool = False,
    blit_numbering: bool = True,
):
    """Perform a single rollout until termination in environment

      :param select_random_single_render:
      :type select_random_single_render:
      :param blit_numbering:
      :type blit_numbering:
      :param rollout_ith:
    :param agent:
    :param drawer:
    :param disable_stdout:
    :param metric_writer:
    :type max_length: int
    :param max_length:
    :type train_agent: bool
    :type render_mode: bool
    :param initial_snapshot: The initial state observation in the environment
    :param env: The environment the agent interacts with
    :param render_mode: Whether to render environment interaction
    :param train_agent: Whether the agent should use the rollout to update its model
    :return:
    -episode_signal (:py:class:`float`) - first output
    -episode_length-
    -average_episode_entropy-"""

    state = agent.extract_features(initial_snapshot)
    running_mean_action = mean_accumulator()
    episode_signal = total_accumulator()

    start = time.time()
    frames = []
    rollout_description = f"Rollout"
    if rollout_ith:
        rollout_description += f" #{rollout_ith}"
    for step_i in progress_bar(
        count(1),
        description=rollout_description,
        unit="th step",
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

        if action is not None:
            running_mean_action.send(action.mean())
        if signal is not None:
            episode_signal.send(signal.mean())

        if render_mode != RenderModeEnum.none:
            frame = env.render(render_mode)
            if frame is not None:
                frames.append(frame)
            if drawer is not None and action is not None:
                if env.action_space.is_singular_discrete:
                    action_a = to_one_hot(agent.output_shape, action)
                else:
                    action_a = action[0]
                drawer.draw(action_a)

        if numpy.array(terminated).all() or (max_length and step_i > max_length):
            break

    fps = step_i / (time.time() - start)
    if train_agent:
        agent.update(metric_writer=metric_writer)
    else:
        logging.info("no update")

    episode_return = next(episode_signal)

    if metric_writer:
        metric_writer.scalar(
            CommonEnvironmentScalarEnum.duration.value, step_i, agent.update_i
        )
        metric_writer.scalar(
            CommonEnvironmentScalarEnum.running_mean_action.value,
            next(running_mean_action),
            agent.update_i,
        )
        metric_writer.scalar(
            CommonEnvironmentScalarEnum.signal.value, episode_return, agent.update_i
        )
        metric_writer.scalar(CommonEnvironmentScalarEnum.fps.value, fps, agent.update_i)
        if (
            isinstance(metric_writer, VideoWriterMixin)
            and render_mode != RenderModeEnum.none
            and render_mode != RenderModeEnum.to_screen
        ):
            video_frames = numpy.array(frames).swapaxes(0, 1)
            input_dims = VideoInputDimsEnum.nthwc
            if select_random_single_render:
                video_frames = video_frames[: random.randint(0, len(video_frames))]

            if blit_numbering:  # also include fps
                fps_writer = int(
                    max(50, fps)
                )  # GIF limit, remove if alternative is used
                video_frames = numpy.array(
                    [
                        blit_fps(blit_numbering_raster_sequence(f), fps_writer)
                        for f in video_frames
                    ]
                )

            metric_writer.video(
                f"{render_mode.value}_replay",
                video_frames,
                step=agent.update_i,
                frame_rate=fps,
                input_dims=input_dims,
            )  # VERY SLOW so do not run to often!

    return episode_return, step_i


class OnPolicyEpisodic(Procedure):
    @passes_kws_to(rollout_on_policy)
    def __call__(
        self,
        *,
        iterations: int = 1000,
        render_frequency: int = 100,
        render_mode: RenderModeEnum = RenderModeEnum.rgb_array,
        stat_frequency: int = 10,
        disable_stdout: bool = False,
        train_agent: bool = True,
        render_next_on_improvement: bool = True,
        metric_writer: Optional[Writer] = MockWriter(),
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
        E = progress_bar(E, description="Rollout #")

        best_episode_return = -math.inf
        render_this_episode = False
        for episode_i in E:
            initial_state = self.environment.reset()

            if render_this_episode:
                mr = metric_writer
                rm = render_mode
            else:
                mr = is_positive_and_mod_zero(
                    stat_frequency, episode_i, ret=metric_writer
                )
                rm = (
                    render_mode
                    if is_positive_and_mod_zero(render_frequency, episode_i)
                    else RenderModeEnum.none
                )
            kwargs.update(render_mode=rm)
            ret, *_ = rollout_on_policy(
                self.agent,
                initial_state,
                self.environment,
                rollout_íth=episode_i,
                metric_writer=mr,
                disable_stdout=disable_stdout,
                **kwargs,
            )
            if train_agent:
                if best_episode_return < ret:
                    best_episode_return = ret
                    self.model_improved(step_i=self.agent.update_i, **kwargs)
                    if render_next_on_improvement:
                        render_this_episode = True
                else:
                    render_this_episode = False

            if self.early_stop:
                break
