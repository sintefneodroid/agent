#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

from agent.exploration.ucb1 import UCB1
from neodroid import  messaging
from neodroid.wrappers import SingleEnvironmentWrapper

__author__ = 'cnheider'


def construct_displayables(normed, tries, totals):
  d1 = messaging.N.Displayable('BeliefBarLeftDisplayer', normed[0])
  d2 = messaging.N.Displayable('BeliefBarMiddleDisplayer', normed[1])
  d3 = messaging.N.Displayable('BeliefBarRightDisplayer', normed[2])
  d12 = messaging.N.Displayable('BeliefTextLeftDisplayer', normed[0])
  d22 = messaging.N.Displayable('BeliefTextMiddleDisplayer', normed[1])
  d32 = messaging.N.Displayable('BeliefTextRightDisplayer', normed[2])
  d13 = messaging.N.Displayable('CountTextLeftDisplayer', f'{totals[0]} / {tries[0]}')
  d23 = messaging.N.Displayable('CountTextMiddleDisplayer', f'{totals[1]} / {tries[1]}')
  d33 = messaging.N.Displayable('CountTextRightDisplayer', f'{totals[2]} / {tries[2]}')
  return [d1, d2, d3, d12, d22, d32, d13, d23, d33]


def main(connect_to_running=False):
  parser = argparse.ArgumentParser(prog='mab')
  parser.add_argument('-C',
                      action='store_true',
                      help='connect to running',
                      default=connect_to_running)
  args = parser.parse_args()
  if args.C:
    connect_to_running = True

  _environment = SingleEnvironmentWrapper(environment_name='mab',
                                          connect_to_running=connect_to_running)

  num_arms = _environment.action_space.num_discrete_actions
  totals = [0] * num_arms

  ucb1 = UCB1(num_arms)

  i = 0
  while _environment.is_connected:
    action = int(ucb1.select_arm())

    i += 1

    action_a = _environment.action_space.ternary_discrete_action_from_idx(action)[0]

    _, signal, terminated, info = _environment.react(action_a).to_gym_like_output()

    ucb1.update_belief(action, signal)

    totals[action] += signal

    _environment.display(displayables=construct_displayables(ucb1.normalised_values,
                                                             ucb1.counts,
                                                             totals))

    if terminated:
      print(info.termination_reason)
      _environment.reset()


if __name__ == '__main__':
  main(connect_to_running=True)
