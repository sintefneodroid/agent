#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys

from neodroid.models import ReactionParameters
from agent.exploration.ucb1 import UCB1

__author__ = 'cnheider'

import neodroid.api_wrappers.single_environment_wrapper as neo
from neodroid import messaging


def main(connect_to_running=False):
  parser = argparse.ArgumentParser(prog='mab')
  parser.add_argument('-C',
                      action='store_true',
                      help='connect to running',
                      default=connect_to_running)
  args = parser.parse_args()
  if args.C:
    connect_to_running = True

  _environment = neo.SingleEnvironmentWrapper(environment_name='mab',
                                              connect_to_running=connect_to_running)

  num_arms = 2

  beliefs = [1 / num_arms] * num_arms
  totals = [0] * num_arms
  tries = [0] * num_arms
  normed = [1 / num_arms] * num_arms

  ucb1 = UCB1(num_arms)

  i = 0
  while _environment.is_connected:
    action = int(ucb1.select_arm())

    print(action)

    a_action = 1
    if action == 0:
      a_action = -1

    i += 1

    _, signal, terminated, info = _environment.react(a_action).to_gym_like_output()

    ucb1.update_belief(action, signal)

    tries[action] += 1
    totals[action] += signal
    beliefs[action] = float(totals[action]) / tries[action]

    for i in range(len(beliefs)):
      normed[i] = beliefs[i] / (sum(beliefs) + sys.float_info.epsilon)

    if terminated:
      print(info.termination_reason)
      _environment.reset()


if __name__ == '__main__':
  main(connect_to_running=True)
