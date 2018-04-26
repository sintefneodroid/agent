#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'
from agents.agent import Agent


class ImitationAgent(Agent):

  def evaluate(self, batch):
    pass

  def sample_action(self, state):
    pass

  def __optimise_wrt__(self, error):
    pass

  def rollout(self, init_obs, env):
    pass
