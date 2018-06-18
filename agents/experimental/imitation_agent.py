#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'
from agents.abstract.agent import Agent


class ImitationAgent(Agent):

  def _build(self):
    pass

  def _defaults(self):
    pass

  def _sample_model(self, state, *args, **kwargs):
    pass

  def update(self, *args, **kwargs):
    pass

  def load(self, *args, **kwargs):
    pass

  def save(self, *args, **kwargs):
    pass

  def __next__(self):
    pass

  def evaluate(self, batch, **kwargs):
    pass

  def sample_action(self, state, **kwargs):
    pass

  def _optimise_wrt(self, error, **kwargs):
    pass

  def rollout(self, init_obs, env, train=True, **kwargs):
    pass
