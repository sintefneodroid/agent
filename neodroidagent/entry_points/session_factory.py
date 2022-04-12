#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/02/2020
           """

import inspect
from os import getenv
from typing import Callable, Type, TypeVar, Union

import torch
from draugr import sprint
from neodroidagent.agents import Agent
from neodroidagent.common.session_factory.vertical.environment_session import (
    EnvironmentSession,
)
from neodroidagent.utilities import NoProcedure
from warg import GDKC, NOD, config_to_mapping

AgentType = TypeVar("AgentType", bound=Agent)
EnvironmentSessionType = TypeVar("EnvironmentSessionType", bound=EnvironmentSession)


def session_factory(
    agent: Type[AgentType] = None,
    config=None,
    *,
    session: Union[Type[EnvironmentSessionType], EnvironmentSession, Callable],
    save: bool = True,
    has_x_server: bool = True,
    skip_confirmation: bool = True,
    **kwargs,
):
    r"""
    Entry point start a starting a training session with the functionality of parsing cmdline arguments and
    confirming configuration to use before training and overwriting of default training configurations"""

    if config is None:
        config = {}

    if isinstance(config, dict):
        config = NOD(**config)
    else:
        config = NOD(config.__dict__)

    if has_x_server:
        display_env = getenv("DISPLAY", None)
        if display_env is None:
            config.RENDER_ENVIRONMENT = False
            has_x_server = False

    config_mapping = config_to_mapping(config)
    config_mapping.update(**kwargs)

    config_mapping.update(save=save, has_x_server=has_x_server)

    if not skip_confirmation:
        sprint(f"\nUsing config: {config}\n", highlight=True, color="yellow")
        for key, arg in config_mapping:
            print(f"{key} = {arg}")

        input("\nPress Enter to begin... ")

    if session is None:
        raise NoProcedure
    elif inspect.isclass(session):
        session = session(**config_mapping)  # Use passed config arguments
    elif isinstance(session, GDKC):
        session = session(
            **kwargs
        )  # Assume some kw parameters is set prior to passing session, only override with explicit overrides

    try:
        session(agent, **config_mapping)
    except KeyboardInterrupt:
        print("Stopping")

    torch.cuda.empty_cache()

    exit(0)
