#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

from neodroid.environments.unity_environment import SingleUnityEnvironment
from neodroid.utilities import Displayable
from neodroidagent.utilities.exploration.ucb1 import UCB1

__author__ = "Christian Heider Nielsen"


def construct_displayables(normed, tries, totals):
    d1 = Displayable("BeliefBarLeftDisplayer", normed[0])
    d2 = Displayable("BeliefBarMiddleDisplayer", normed[1])
    d3 = Displayable("BeliefBarRightDisplayer", normed[2])
    d12 = Displayable("BeliefTextLeftDisplayer", normed[0])
    d22 = Displayable("BeliefTextMiddleDisplayer", normed[1])
    d32 = Displayable("BeliefTextRightDisplayer", normed[2])
    d13 = Displayable("CountTextLeftDisplayer", f"{totals[0]} / {tries[0]}")
    d23 = Displayable("CountTextMiddleDisplayer", f"{totals[1]} / {tries[1]}")
    d33 = Displayable("CountTextRightDisplayer", f"{totals[2]} / {tries[2]}")
    return [d1, d2, d3, d12, d22, d32, d13, d23, d33]


def main(connect_to_running=True):
    parser = argparse.ArgumentParser(prog="mab")
    parser.add_argument(
        "-C", action="store_true", help="connect to running", default=connect_to_running
    )
    args = parser.parse_args()
    if args.C:
        connect_to_running = connect_to_running

    _environment = SingleUnityEnvironment(
        environment_name="mab", connect_to_running=connect_to_running
    )

    num_arms = _environment.action_space.discrete_steps
    totals = [0] * num_arms
    ucb1 = UCB1(num_arms)

    while _environment.is_connected:
        action = ucb1.select_arm()

        a = _environment.react(action)
        signal, terminated = a.signal, a.terminated
        ucb1.update_belief(action, signal)
        totals[action] += signal

        _environment.display(
            displayables=construct_displayables(
                ucb1.normalised_values, ucb1.counts, totals
            )
        )

        if terminated:
            _environment.reset()


if __name__ == "__main__":
    main(connect_to_running=True)
