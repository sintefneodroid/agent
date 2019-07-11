#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC

from agent.interfaces.torch_agent import TorchAgent

__author__ = 'cnheider'


class ImitationAgent(TorchAgent, ABC):

  # region Private

  def __next__(self):
    pass

  # endregion

  # region Public

  def update(self, *args, **kwargs):
    pass

  def load(self, *args, **kwargs):
    pass

  def save(self, *args, **kwargs):
    pass

  def evaluate(self, batch, **kwargs):
    pass

  def sample(self, state, **kwargs):
    pass

  def rollout(self, init_obs, env, train=True, **kwargs):
    pass

  # endregion

  # region Protected

  def _build(self, env, **kwargs) -> None:
    pass

  def __defaults__(self) -> None:
    pass

  def _sample_model(self, state, **kwargs):
    pass

  def _optimise_wrt(self, error, **kwargs):
    pass

  # endregion
