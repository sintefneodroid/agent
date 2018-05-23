#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

from agents.experimental.evo_agent import EVOAgent


class GAAgent(EVOAgent):

  def evaluate(self, batch, **kwargs):
    pass

  def __optimise_wrt__(self, error, **kwargs):
    pass

  def rollout(self, init_obs, env, **kwargs):
    pass

  def sample_action(self, state, **kwargs):
    pass
