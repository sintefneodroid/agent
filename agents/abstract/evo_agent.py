#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC

__author__ = 'cnheider'
from agents.abstract.agent import Agent


class EVOAgent(Agent, ABC):

  # region Public

  def sample_action(self, state, **kwargs):
    pass

  def evaluate(self, batch, **kwargs):
    pass

  def rollout(self, init_obs, env, train=True, **kwargs):
    pass

  # endregion

  # region Protected

  def _optimise_wrt(self, error, **kwargs):
    pass

  # endregion
