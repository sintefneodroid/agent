#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'
from agents.agent import Agent


class EVOAgent(Agent):

  def sample_action(self, state, **kwargs):
    pass

  def __optimise_wrt__(self, error, **kwargs):
    pass

  def evaluate(self, batch, **kwargs):
    pass

  def rollout(self, init_obs, env, **kwargs):
    pass
