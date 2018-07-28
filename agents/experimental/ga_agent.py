#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

from agents.abstract.evo_agent import EVOAgent


class GAAgent(EVOAgent):

  def evaluate(self, batch, **kwargs):
    pass

  def _optimise_wrt(self, error, **kwargs):
    pass

  def rollout(self, init_obs, env, train=True, **kwargs):
    pass

  def sample_action(self, state, **kwargs):
    pass
