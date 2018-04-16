#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'


import utilities as U
from neodroid.wrappers.curriculum_wrapper import NeodroidCurriculumWrapper
from neodroid.wrappers.gym_wrapper import NeodroidGymWrapper


class BinaryActionEncodingWrapper(NeodroidGymWrapper):

  def step(self, a, *args, **kwargs):
    a = U.signed_one_hot_encoding(self.action_space.num_actions, a)
    return super().step(a)


class BinaryActionEnvironment(NeodroidCurriculumWrapper):

  def step(self, a, *args, **kwargs):
    a = U.signed_one_hot_encoding(self.action_space.num_actions, a)
    return super().act(a)
