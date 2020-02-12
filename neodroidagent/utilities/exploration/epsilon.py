#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """

__all__ = ["exploration_action"]


def exploration_action(agent, state):
    """
choose an action based on state with random noise added for exploration in training

:param agent:
:param state:
:return:
"""

    softmax_action = agent._sample_model(state)
    epsilon = agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * numpy.exp(
        -1.0 * agent._step_i / agent.epsilon_decay
    )
    if numpy.random.rand() < epsilon:
        action = numpy.random.choice(agent.action_dim)
    else:
        action = numpy.argmax(softmax_action)
    return action
