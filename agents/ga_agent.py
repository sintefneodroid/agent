#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'
from agents.agent import Agent


class GAAgent(Agent):
  def __init__(self, config):
    super().__init__()
    self._use_cuda = config.USE_CUDA_IF_AVAILABLE
    self._step_n = 0
    self._rollout_i = 0
    self.C = config
