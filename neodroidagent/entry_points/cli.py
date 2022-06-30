#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 19/01/2020
           """

import draugr
import fire
from pyfiglet import Figlet
from warg import NOD
from warg.arguments import upper_dict

from neodroidagent import get_version
from neodroidagent.entry_points.agent_tests import AGENT_CONFIG, AGENT_OPTIONS

margin_percentage = 0 / 6
terminal_width = draugr.get_terminal_size().columns
margin = int(margin_percentage * terminal_width)
width = terminal_width - 2 * margin
underline = "_" * width
indent = " " * margin
sponsors = ("SINTEF Ocean", "Alexandra Institute", "Norges Forskningsråd")


class RunAgent(object):
    """ """

    def __init__(self, agent_key: str, agent_callable: callable):
        self.agent_key = agent_key
        self.agent_callable = agent_callable

    def train(self, **explicit_overrides) -> None:
        """

        :param explicit_overrides: Accepts kwarg overrides to config
        :return:"""
        default_config = NOD(AGENT_CONFIG[self.agent_key])

        config_overrides = upper_dict(explicit_overrides)
        for key, arg in config_overrides.items():
            setattr(default_config, key, arg)

        print(f"Explicit Overrides:\n{explicit_overrides}")

        self.agent_callable(config=default_config, **explicit_overrides)

    def run(self):
        self.train(
            train_agent=False,
            render_frequency=1,
            save=False,
            save_ending=False,
            num_envs=1,
            save_best_throughout_training=False,
        )


class NeodroidAgentCLI:
    """ """

    def __init__(self):
        for k, v in AGENT_OPTIONS.items():
            setattr(self, k, RunAgent(k, v))

    @staticmethod
    def version() -> None:
        """
        Prints the version of this Neodroid installation."""
        draw_cli_header()
        print(f"Version: {get_version()}")

    @staticmethod
    def sponsors() -> None:
        print(sponsors)


def draw_cli_header(*, title: str = "Neodroid Agent", font: str = "big") -> None:
    figlet = Figlet(font=font, justify="center", width=terminal_width)
    description = figlet.renderText(title)

    print(f"{description}{underline}\n")


def main(*, always_draw_header: bool = False) -> None:
    if always_draw_header:
        draw_cli_header()
    fire.Fire(NeodroidAgentCLI, name="neodroid-agent")


if __name__ == "__main__":
    main()
