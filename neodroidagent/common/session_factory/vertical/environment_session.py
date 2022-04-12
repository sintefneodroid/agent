#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
from enum import Enum
from typing import Type, TypeVar, Union

from neodroid.environments import Environment
from neodroidagent.agents import Agent

from .procedures import OnPolicyEpisodic, Procedure

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""

__all__ = ["EnvironmentSession"]

ProcedureType = TypeVar("ProcedureType", bound=Procedure)


class EnvironmentType(Enum):
    gym = "gym"
    zmq_pipe = "zmq_pipe"
    unity = "unity"
    blender = "blender"


class EnvironmentSession(abc.ABC):
    def __init__(
        self,
        *,
        environment: Union[Environment, EnvironmentType],
        procedure: Union[Type[ProcedureType], Procedure] = OnPolicyEpisodic,
        **kwargs,
    ):
        self._environment = environment
        self._procedure = procedure

    @abc.abstractmethod
    def __call__(
        self,
        agent: Type[Agent],
        *,
        load_time,
        seed,
        save_model: bool = True,
        load_previous_environment_model_if_available: bool = False,
        train_agent=True,
        **kwargs,
    ):
        raise NotImplementedError
