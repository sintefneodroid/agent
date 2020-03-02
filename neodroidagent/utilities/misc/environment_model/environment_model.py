#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27/02/2020
           """

from collections import defaultdict

import numpy as np


class EnvModel(object):
    """
  A simple tabular environment model that maintains the counts of each
  reward-outcome pair given the state and action that preceded them. The
  model can be queried with

  >>> M = EnvModel()
  >>> M[(state, action, reward, next_state)] += 1
  >>> M[(state, action, reward, next_state)]
  1
  >>> M.state_action_pairs()
  [(state, action)]
  >>> M.outcome_probs(state, action)
  [(next_state, 1)]
  """

    def __init__(self):
        super(EnvModel, self).__init__()
        self._model = defaultdict(lambda: defaultdict(lambda: 0))

    def __setitem__(self, key, value):
        s, a, r, s_ = key
        self._model[(s, a)][(r, s_)] = value

    def __getitem__(self, key):
        s, a, r, s_ = key
        return self._model[(s, a)][(r, s_)]

    def __contains__(self, key):
        s, a, r, s_ = key
        p1 = (s, a) in self.state_action_pairs()
        p2 = (r, s_) in self.reward_outcome_pairs()
        return p1 and p2

    def state_action_pairs(self):
        """
Return all (state, action) pairs in the environment model
"""
        return list(self._model.keys())

    def reward_outcome_pairs(self, s, a):
        """
Return all (reward, next_state) pairs associated with taking action `a`
in state `s`.
"""
        return list(self._model[(s, a)].keys())

    def outcome_probs(self, s, a):
        """
Return the probability under the environment model of each outcome
state after taking action `a` in state `s`.

Parameters
----------
s : int as returned by ``self._obs2num``
The id for the state/observation.
a : int as returned by ``self._action2num``
The id for the action taken from state `s`.

Returns
-------
outcome_probs : list of (state, prob) tuples
A list of each possible outcome and its associated probability
under the model.
"""
        items = list(self._model[(s, a)].items())
        total_count = np.sum([c for (_, c) in items])
        outcome_probs = [c / total_count for (_, c) in items]
        outcomes = [p for (p, _) in items]
        return list(zip(outcomes, outcome_probs))

    def state_action_pairs_leading_to_outcome(self, outcome):
        """
Return all (state, action) pairs that have a nonzero probability of
producing `outcome` under the current model.

Parameters
----------
outcome : int
The outcome state.

Returns
-------
pairs : list of (state, action) tuples
A list of all (state, action) pairs with a nonzero probability of
producing `outcome` under the model.
"""
        pairs = []
        for sa in self.state_action_pairs():
            outcomes = [o for (r, o) in self.reward_outcome_pairs(*sa)]
            if outcome in outcomes:
                pairs.append(sa)
        return pairs
