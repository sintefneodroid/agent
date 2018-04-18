#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'

from agents.experimental.evo_agent import EVOAgent


class GAAgent(EVOAgent):
  def evaluate(self, batch):
    pass

  def optimise_wrt(self, error):
    pass

  def rollout(self, init_obs, env):
    pass

  def sample_action(self, state):
    pass
