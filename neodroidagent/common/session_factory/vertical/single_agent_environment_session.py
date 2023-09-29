#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""

__all__ = ["SingleAgentEnvironmentSession"]

import inspect
import time
from os import cpu_count
from typing import Any, Type

import torch
from draugr.drawers import DiscreteScrollPlot, SeriesScrollPlot
from draugr.random_utilities import seed_stack
from draugr.stopping import (
    CaptureEarlyStop,
)
from draugr.torch_utilities import TensorBoardPytorchWriter
from draugr.writers import MockWriter
from neodroidagent import PROJECT_APP_PATH
from neodroidagent.agents import Agent
from neodroidagent.utilities import NoAgent
from warg import GDKC, passes_kws_to

# import torchsnooper
from warg import IgnoreInterruptSignal
from draugr.python_utilities import sprint
from warg.context_wrapper import ContextWrapper
from warg.decorators.timing import StopWatch

from .environment_session import EnvironmentSession
from .procedures.procedure_specification import Procedure, DrawingModeEnum


class SingleAgentEnvironmentSession(EnvironmentSession):
    """
    Description
    """

    @passes_kws_to(
        CaptureEarlyStop.__init__,
        Agent.__init__,
        Agent.save,
        Procedure.__init__,
        Procedure.__call__,
    )
    def __call__(
        self,
        agent: Type[Agent],
        *,
        load_time: Any = str(int(time.time())),
        seed: int = 0,
        save_ending_model: bool = False,
        save_training_resume: bool = False,
        load_previous_model_if_available: bool = False,
        train_agent: bool = True,
        debug: bool = False,
        num_envs: int = cpu_count() // 3,
        drawing_mode: DrawingModeEnum = DrawingModeEnum.all,
        insist_metric_logging: bool = False,
        **kwargs,
    ) -> None:
        """

        Start a session, builds Agent and starts/connect environment(s), and runs Procedure

        :param agent:
        :type agent:
        :param load_time:
        :type load_time:
        :param seed:
        :type seed:
        :param save_ending_model:
        :type save_ending_model:
        :param save_training_resume:
        :type save_training_resume:
        :param load_previous_model_if_available:
        :type load_previous_model_if_available:
        :param train_agent:
        :type train_agent:
        :param debug:
        :type debug:
        :param num_envs:
        :type num_envs:
        :param drawing_mode:
        :type drawing_mode:
        :param insist_metric_logging:
        :type insist_metric_logging:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        kwargs.update(num_envs=num_envs)
        kwargs.update(train_agent=train_agent)
        kwargs.update(debug=debug)
        kwargs.update(environment=self._environment)

        # with ContextWrapper(torchsnooper.snoop, debug):
        if True:
            with ContextWrapper(torch.autograd.detect_anomaly, debug):
                if agent is None:
                    raise NoAgent

                if inspect.isclass(agent):
                    sprint(
                        "Instantiating Agent", color="crimson", bold=True, italic=True
                    )
                    seed_stack(seed)
                    self._environment.seed(seed)

                    agent = agent(load_time=load_time, seed=seed, **kwargs)

                agent_class_name = agent.__class__.__name__

                total_shape = "_".join(
                    [
                        str(i)
                        for i in (
                            self._environment.observation_space.shape
                            + self._environment.action_space.shape
                            + self._environment.signal_space.shape
                        )
                    ]
                )

                environment_name = f"{self._environment.environment_name}_{total_shape}"

                save_directory = (
                    PROJECT_APP_PATH.user_data / environment_name / agent_class_name
                )
                log_directory = (
                    PROJECT_APP_PATH.user_log
                    / environment_name
                    / agent_class_name
                    / load_time
                )

                if (
                    self._environment.action_space.is_singular_discrete
                    and drawing_mode == DrawingModeEnum.actions
                ):
                    drawer_type = GDKC(
                        DiscreteScrollPlot,
                        num_bins=self._environment.action_space.discrete_steps,
                        default_delta=None,
                    )
                else:
                    drawer_type = GDKC(
                        SeriesScrollPlot, window_length=100, default_delta=None
                    )

                if train_agent or insist_metric_logging:
                    metric_writer = GDKC(TensorBoardPytorchWriter, path=log_directory)
                else:
                    metric_writer = GDKC(MockWriter)

                with ContextWrapper(metric_writer, train_agent) as metric_writer:
                    with ContextWrapper(
                        drawer_type,
                        not train_agent and drawing_mode != DrawingModeEnum.none,
                    ) as drawer_instance:
                        agent.build(
                            self._environment.observation_space,
                            self._environment.action_space,
                            self._environment.signal_space,
                            metric_writer=metric_writer,
                        )

                        kwargs.update(
                            environment_name=(self._environment.environment_name,),
                            save_directory=save_directory,
                            log_directory=log_directory,
                            load_time=load_time,
                            seed=seed,
                            train_agent=train_agent,
                        )

                        found = False
                        if load_previous_model_if_available:
                            sprint(
                                "Searching for previously trained models for initialisation for this configuration "
                                "(Architecture, Action Space, Observation Space, ...)",
                                color="crimson",
                                bold=True,
                                italic=True,
                            )
                            found = agent.load(
                                save_directory=save_directory,
                                evaluation=not train_agent,
                            )
                            if not found:
                                sprint(
                                    "Did not find any previously trained models for this configuration",
                                    color="crimson",
                                    bold=True,
                                    italic=True,
                                )

                        if not train_agent:
                            agent.eval()
                        else:
                            agent.train()

                        if not found:
                            sprint(
                                "Training from new initialisation",
                                color="crimson",
                                bold=True,
                                italic=True,
                            )

                        cbs = []
                        if metric_writer:
                            cbs.append(
                                lambda *_, step_i, **__: metric_writer.blip(
                                    "new_best_model", step_i
                                )
                            )
                        session_proc = self._procedure(
                            agent,
                            on_improvement_callbacks=cbs,
                            **kwargs,
                        )

                        with CaptureEarlyStop(
                            self._procedure.stop_procedure, combinations=[], **kwargs
                        ):
                            with StopWatch() as timer:
                                with IgnoreInterruptSignal():
                                    training_resume = session_proc(
                                        metric_writer=metric_writer,
                                        drawer=drawer_instance,
                                        **kwargs,
                                    )
                                    if (
                                        training_resume
                                        and "stats" in training_resume
                                        and save_training_resume
                                    ):
                                        training_resume.stats.save(**kwargs)

                        session_proc.close()

                        end_message = f"Training ended, time elapsed: {timer // 60:.0f}m {timer % 60:.0f}s"
                        line_width = 9
                        sprint(
                            f'\n{"-" * line_width} {end_message} {"-" * line_width}\n',
                            color="crimson",
                            bold=True,
                            italic=True,
                        )

                        if save_ending_model:
                            agent.save(**kwargs)

                try:
                    self._environment.close()
                except BrokenPipeError:
                    pass

                exit(0)


if __name__ == "__main__":
    print(SingleAgentEnvironmentSession)
