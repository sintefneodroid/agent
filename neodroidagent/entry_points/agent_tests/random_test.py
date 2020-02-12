#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/01/2020
           """

from typing import Union

from neodroidagent.agents import RandomAgent
from neodroidagent.common import ParallelSession
from neodroidagent.entry_points.session_factory import session_factory

random_config = globals()


def random_run(
    rollouts=None,
    skip_confirmation: bool = True,
    environment_type: Union[bool, str] = True,
    config=random_config,
) -> None:
    if rollouts:
        config.ROLLOUTS = rollouts

    session_factory(
        RandomAgent,
        config,
        session=ParallelSession,
        skip_confirmation=skip_confirmation,
        environment_type=environment_type,
    )


def random_test(config=random_config) -> None:
    random_run(environment_type="gym", config=config)


if __name__ == "__main__":
    random_test()
