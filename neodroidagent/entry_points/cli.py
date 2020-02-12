#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 19/01/2020
           """

import fire
from pyfiglet import Figlet

import draugr
from neodroidagent import get_version
from neodroidagent.entry_points.agent_tests import AGENT_CONFIG, AGENT_OPTIONS
from warg import NOD
from warg.arguments import upper_dict

margin_percentage = 0 / 6
terminal_width = draugr.get_terminal_size().columns
margin = int(margin_percentage * terminal_width)
width = terminal_width - 2 * margin
underline = "_" * width
indent = " " * margin
sponsors = "SINTEF Ocean, Alexandra Institute, Norges Forskningsråd"


class RunAgent:
    def __init__(self, agent_key, agent_callable):
        self.agent_key = agent_key
        self.agent_callable = agent_callable

    def train(self, **overrides):
        """

    @param overrides:
    @return:
    """
        default_config = NOD(AGENT_CONFIG[self.agent_key])

        overrides = upper_dict(overrides)
        for key, arg in overrides.items():
            setattr(default_config, key, arg)

        print("Overrides:")
        print(overrides)
        print(default_config)

        self.agent_callable(config=default_config)

    def run(self):
        pass


class NeodroidAgentCLI:
    def __init__(self):
        for k, v in AGENT_OPTIONS.items():
            setattr(self, k, RunAgent(k, v))

    @staticmethod
    def version() -> None:
        """
    Prints the version of this Neodroid installation.
    """
        draw_cli_header()
        print(f"Version: {get_version()}")

    @staticmethod
    def sponsors() -> None:
        print(sponsors)


def draw_cli_header(*, title="Neodroid Agent", font="big"):
    figlet = Figlet(font=font, justify="center", width=terminal_width)
    description = figlet.renderText(title)

    print(f"{description}{underline}\n")


def main(*, always_draw_header=False):
    if always_draw_header:
        draw_cli_header()
    fire.Fire(NeodroidAgentCLI, name="neodroid-agent")


if __name__ == "__main__":
    main()
