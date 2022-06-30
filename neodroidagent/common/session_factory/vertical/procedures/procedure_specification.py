#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
from pathlib import Path
from typing import Iterable, List, Union

from draugr import min_interval_wrapper_global
from sorcery import assigned_names
from warg import drop_unused_kws

from neodroid.environments import Environment
from neodroidagent.agents import Agent

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""
__all__ = ["Procedure"]


class DrawingModeEnum:
    (
        none,
        actions,  # actions single environment actions
        actions_all,  # actions all environment interactions
        states,  # actions single environment states,
        states_all,  # actions all environment states
        all,  # TODO: not supported yet
    ) = assigned_names()


class Procedure(abc.ABC):
    early_stop = False  # End Training flag

    @drop_unused_kws
    def __init__(
        self,
        agent: Agent,
        *,
        environment: Environment,
        on_improvement_callbacks=None,
        save_best_throughout_training: bool = True,
        train_agent: bool = True,
        min_save_interval: int = 0
    ):
        """

        :param agent:
        :param environment:
        :param on_improvement_callbacks:
        :param save_best_throughout_training:"""
        if not isinstance(on_improvement_callbacks, List) and isinstance(
            on_improvement_callbacks, Iterable
        ):
            on_improvement_callbacks = [*on_improvement_callbacks]
        elif on_improvement_callbacks is None:
            on_improvement_callbacks = []

        self.agent = agent
        self.environment = environment
        if save_best_throughout_training and train_agent:
            func = self.agent.save
            if min_save_interval:
                min_interval_wrapper_global(func, min_interval=min_save_interval)
            on_improvement_callbacks.append(func)
        self.on_improvement_callbacks = on_improvement_callbacks

    @staticmethod
    def stop_procedure() -> None:
        """

        :return:"""
        print("STOPPING PROCEDURE!")
        Procedure.early_stop = True

    def model_improved(self, *, step_i, verbose: bool = True, **kwargs):
        """

        :param step_i:
        :type step_i:
        :param verbose:
        :param kwargs:
        :return:"""
        if verbose:
            print("Model improved")

        [
            cb(step_i=step_i, verbose=verbose, **kwargs)
            for cb in self.on_improvement_callbacks
        ]


@abc.abstractmethod
def __call__(
    self,
    *,
    iterations: int = 9999,
    log_directory: Union[str, Path],
    render_frequency: int = 100,
    stat_frequency: int = 10,
    disable_stdout: bool = False,
    train_agent: bool = True,
    **kwargs
):
    """
    Collects environment snapshots and forwards it to the agent and vice versa.

    :param agent:
    :param environment:
    :param num_steps_per_btach:
    :param num_updates:
    :param iterations:
    :param log_directory:
    :param render_frequency:
    :param stat_frequency:
    :param kwargs:
    :return:"""
    raise NotImplementedError


def close(self):
    self.environment.close()
