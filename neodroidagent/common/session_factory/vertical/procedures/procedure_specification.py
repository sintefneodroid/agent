#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
from pathlib import Path
from typing import Union, List

from neodroid import Environment
from neodroidagent.agents import Agent
from warg import drop_unused_kws

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""
__all__ = ["Procedure"]


class Procedure(abc.ABC):
    early_stop = False  # End Training flag

    @drop_unused_kws
    def __init__(
        self,
        agent: Agent,
        *,
        environment: Environment,
        on_improvement_callbacks: List = [],
        save_best_throughtout_training: bool = True
    ):
        """

    @param agent:
    @param environment:
    @param on_improvement_callbacks:
    @param save_best_throughtout_training:
    """
        self.agent = agent
        self.environment = environment
        if save_best_throughtout_training:
            on_improvement_callbacks.append(self.agent.save)
        self.on_improvement_callbacks = on_improvement_callbacks

    def stop_procedure(self) -> None:
        """

    @return:
    """
        self.early_stop = True

    def call_on_improvement_callbacks(self, *, verbose: bool = True, **kwargs):
        """

    @param verbose:
    @param kwargs:
    @return:
    """
        if verbose:
            print("Model improved")

        if self.on_improvement_callbacks:
            [cb(verbose=verbose, **kwargs) for cb in self.on_improvement_callbacks]

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
    :return:
    """
        raise NotImplementedError
