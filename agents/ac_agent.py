#!/usr/bin/env python3
# coding=utf-8
__author__='cnheider'
from agents.agent import Agent


class ACAgent(Agent):
  """
  All value iteration agents should inherit from this class
  """

  def __init__(self):
    super().__init__()

  def forward(self, state, *args, **kwargs):
    raise NotImplementedError()
