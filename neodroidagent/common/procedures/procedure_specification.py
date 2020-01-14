#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
from pathlib import Path
from typing import Union

from neodroid.environments.unity_environment.vector_unity_environment import (
    VectorUnityEnvironment,
)
from neodroidagent import Agent
from warg import drop_unused_kws

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""
__all__ = ["Procedure"]


class Procedure(abc.ABC):
    early_stop = False  # End Training flag

    @drop_unused_kws
    def __init__(self, agent: Agent, *, environment: VectorUnityEnvironment):
        self.agent = agent
        self.environment = environment

    def stop_procedure(self) -> None:
        self.early_stop = True

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
