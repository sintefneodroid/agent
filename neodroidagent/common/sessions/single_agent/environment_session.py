#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
import inspect
import time
from typing import Type, TypeVar, Union

from draugr.stopping.stopping_key import add_early_stopping_key_combination
from draugr.torch_utilities import torch_seed
from neodroid.environments.environment import Environment
from neodroidagent import PROJECT_APP_PATH
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.utilities.exceptions.exceptions import NoAgent
from neodroidagent.common.procedures.training.on_policy_episodic import OnPolicyEpisodic
from neodroidagent.common.procedures.procedure_specification import Procedure
from warg import passes_kws_to

__author__ = "Christian Heider Nielsen"
__doc__ = ""

ProcedureType = TypeVar("ProcedureType", bound=Procedure)

__all__ = ["EnvironmentSession"]


class EnvironmentSession(abc.ABC):
    def __init__(
        self,
        *,
        environments: Environment,
        procedure: Union[Type[ProcedureType], Procedure] = OnPolicyEpisodic,
        **kwargs,
    ):
        self._environment = environments
        self._procedure = procedure

    @passes_kws_to(
        add_early_stopping_key_combination,
        TorchAgent.__init__,
        TorchAgent.save,
        Procedure.__call__,
    )
    def __call__(
        self,
        agent: Type[TorchAgent],
        *,
        load_time,
        seed,
        save_model: bool = True,
        load_previous_environment_model_if_available: bool = False,
        train_agent=True,
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
            [m.eval() for m in agent.models.values()]

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

        model_directory = (
            PROJECT_APP_PATH.user_data / environment_name / agent_class_name
        )
        log_directory = (
            PROJECT_APP_PATH.user_log / environment_name / agent_class_name / load_time
        )

        if load_previous_environment_model_if_available:
            print("Searching for previous trained models for initialisation")
            agent.load(model_path=model_directory, evaluation=False)

        listener = add_early_stopping_key_combination(
            self._procedure.stop_procedure, **kwargs
        )

        proc = self._procedure(agent, environment=self._environment)

        training_start_timestamp = time.time()
        if listener:
            listener.start()

        kwargs["environment_name"] = (self._environment.environment_name,)

        try:
            training_resume = proc(
                log_directory=log_directory,
                load_time=load_time,
                seed=seed,
                train_agent=train_agent,
                **kwargs,
            )
            if training_resume and "stats" in training_resume:
                training_resume.stats.save(log_directory=log_directory, **kwargs)

        except KeyboardInterrupt:
            pass
        time_elapsed = time.time() - training_start_timestamp

        if listener:
            listener.stop()

        end_message = f"Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        line_width = 9
        print(f'\n{"-" * line_width} {end_message} {"-" * line_width}\n')

        if save_model:
            agent.save(save_directory=model_directory, **kwargs)

        try:
            self._environment.close()
        except BrokenPipeError:
            pass

        exit(0)


if __name__ == "__main__":
    print(EnvironmentSession)
