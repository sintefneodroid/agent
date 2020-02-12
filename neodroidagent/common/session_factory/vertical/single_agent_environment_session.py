#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import inspect
import time
from typing import TypeVar, Type, Any

from draugr import add_early_stopping_key_combination, torch_seed
from neodroidagent import PROJECT_APP_PATH
from neodroidagent.agents import Agent
from .procedures.procedure_specification import Procedure
from .environment_session import EnvironmentSession
from neodroidagent.utilities import NoAgent
from warg import passes_kws_to

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""

__all__ = ["SingleAgentEnvironmentSession"]

ProcedureType = TypeVar("ProcedureType", bound=Procedure)


class SingleAgentEnvironmentSession(EnvironmentSession):
    @passes_kws_to(
        add_early_stopping_key_combination,
        Agent.__init__,
        Agent.save,
        Procedure.__call__,
    )
    def __call__(
        self,
        agent: Type[Agent],
        *,
        load_time: Any,
        seed: int,
        save_model: bool = True,
        continue_training: bool = True,
        train_agent: bool = True,
        **kwargs,
    ):
        """
    Start a session, builds Agent and starts/connect environment(s), and runs Procedure


    :param args:
    :param kwargs:
    :return:
    """

        if agent is None:
            raise NoAgent

        if inspect.isclass(agent):
            torch_seed(seed)
            self._environment.seed(seed)

            agent = agent(load_time=load_time, seed=seed, **kwargs)
            agent.build(
                self._environment.observation_space,
                self._environment.action_space,
                self._environment.signal_space,
            )

        if not train_agent:
            agent.eval()

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
            PROJECT_APP_PATH.user_log / environment_name / agent_class_name / load_time
        )

        kwargs.update(
            environment_name=(self._environment.environment_name,),
            save_directory=save_directory,
            log_directory=log_directory,
            load_time=load_time,
            seed=seed,
            train_agent=train_agent,
        )

        if continue_training:
            print("Searching for previous trained models for initialisation")
            agent.load(save_directory=save_directory, evaluation=not train_agent)
        else:
            print("Training from new initialisation")

        listener = add_early_stopping_key_combination(
            self._procedure.stop_procedure, **kwargs
        )

        proc = self._procedure(agent, environment=self._environment)

        training_start_timestamp = time.time()
        if listener:
            listener.start()

        try:
            training_resume = proc(**kwargs)
            if training_resume and "stats" in training_resume:
                training_resume.stats.save(**kwargs)

        except KeyboardInterrupt:
            pass
        time_elapsed = time.time() - training_start_timestamp

        if listener:
            listener.stop()

        end_message = f"Training ended, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        line_width = 9
        print(f'\n{"-" * line_width} {end_message} {"-" * line_width}\n')

        if save_model:
            agent.save(**kwargs)

        try:
            self._environment.close()
        except BrokenPipeError:
            pass

        exit(0)


if __name__ == "__main__":
    print(SingleAgentEnvironmentSession)
