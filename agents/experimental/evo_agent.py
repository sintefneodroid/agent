#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'
from agents.agent import Agent


class EVOAgent(Agent):
  def sample_action(self, state):
    pass

  def optimise_wrt(self, error):
    pass

  def evaluate(self, batch):
    pass

  def rollout(self, init_obs, env):
    pass
