#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'
import torch

import utilities as U
from agents.abstract.agent import Agent


class JointACAgent(Agent):
  '''
All value iteration agents should inherit from this class
'''

  def __init__(self, *args, **kwargs):
    self._actor_critic_arch = None
    self._value_arch_parameters = None
    self._actor_critic = None

    super().__init__(*args, **kwargs)

  def save(self, C):
    U.save_model(self._actor_critic, C)

  def load(
      self, model_path, evaluation=False
      ):  # TODO: do not use _model as model
    print('Loading latest model: ' + model_path)
    self._actor_critic = self._actor_critic_arch(**self._value_arch_parameters)
    self._actor_critic.load_state_dict(torch.load(model_path))
    if evaluation:
      self._actor_critic = self._actor_critic.eval()
      self._actor_critic.train(False)
    if self._use_cuda:
      self._actor_critic = self._actor_critic.cuda()
    else:
      self._actor_critic = self._actor_critic.cpu()
