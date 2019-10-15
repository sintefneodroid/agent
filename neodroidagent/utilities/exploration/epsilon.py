#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 09/10/2019
           '''


def exploration_action(self, state):
  '''
  choose an action based on state with random noise added for exploration in training

  :param self:
  :param state:
  :return:
  '''

  softmax_action = self._sample_model(state)
  epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * numpy.exp(
    -1.0 * self._step_i / self.epsilon_decay
    )
  if numpy.random.rand() < epsilon:
    action = numpy.random.choice(self.action_dim)
  else:
    action = numpy.argmax(softmax_action)
  return action
