#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest

from neodroidagent.utilities.signal.experimental.generalised_advantage import discounted_gae, discounted_ge
from neodroidagent.utilities.signal.experimental.nstep import discounted_nstep, discounted_nstep_adv
from warg.named_ordered_dictionary import NOD

__author__ = 'cnheider'
__doc__ = ''

import numpy as np


def sample_transitions():
  signals = np.array([[1, 2, 3, 4, 5],
                      [5, 4, 3, 2, 1]
                      ],
                     np.float32)
  terminal = np.array([[0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0]
                       ],
                      np.float32)
  values = np.array([[-100, 10, 20, 30, 40, 50],
                     [-150, 15, 25, 35, 45, 55]
                     ],
                    np.float32)  # Future values

  return NOD.nod_of(signals, terminal, values)


@pytest.fixture
def transitions():
  return sample_transitions()


def test_discounted_gae_returns(transitions):
  # given
  s = transitions.signals
  t = transitions.terminal
  v = transitions.values
  steps = 0.9
  d = 0.8
  # when
  actual = discounted_gae(signals=s, values=v, terminals=t, discount_factor=d, step_factor=steps)
  # then
  expected = np.array([
    [(s[0, 0] + d * v[0, 1] - v[0, 0]) + d * steps * (
        (s[0, 1] + d * v[0, 2] - v[0, 1]) + d * steps * (s[0, 2] - v[0, 2])
    ),
     (s[0, 1] + d * v[0, 2] - v[0, 1]) + d * steps * (s[0, 2] - v[0, 2]),
     (s[0, 2] - v[0, 2]),
     (s[0, 3] + d * v[0, 4] - v[0, 3]) + d * steps * (s[0, 4] + d * v[0, 5] - v[0, 4]),
     (s[0, 4] + d * v[0, 5] - v[0, 4]),
     ],

    [(s[1, 0] + d * v[1, 1] - v[1, 0]) + d * steps * ((s[1, 1] + d * v[1, 2] - v[1, 1]) + d * steps * (
        (s[1, 2] + d * v[1, 3] - v[1, 2]) + d * steps * (
        (s[1, 3] + d * v[1, 4] - v[1, 3]) + d * steps * (s[1, 4] + d * v[1, 5] - v[1, 4])))),
     (s[1, 1] + d * v[1, 2] - v[1, 1]) + d * steps * (
         (s[1, 2] + d * v[1, 3] - v[1, 2]) + d * steps * ((s[1, 3] + d * v[1, 4] - v[1, 3]) + d * steps * (
         s[1, 4] + d * v[1, 5] - v[1, 4]))),
     (s[1, 2] + d * v[1, 3] - v[1, 2]) + d * steps * (
         (s[1, 3] + d * v[1, 4] - v[1, 3]) + d * steps * (s[1, 4] + d * v[1, 5] - v[1, 4])),
     (s[1, 3] + d * v[1, 4] - v[1, 3]) + d * steps * (s[1, 4] + d * v[1, 5] - v[1, 4]),
     (s[1, 4] + d * v[1, 5] - v[1, 4])
     ]
    ])
  np.testing.assert_allclose(expected, actual)


def test_n_step_advantage_returns(transitions):
  # given
  s = transitions.signals
  t = transitions.terminal
  v = transitions.values

  d = 0.9
  n_step = 4
  # when
  actual = discounted_nstep_adv(s, v, t, discount_factor=d, n=n_step)
  # then
  expected = np.array([
    [s[0, 0] + d * (s[0, 1] + d * s[0, 2]) - v[0, 0],
     s[0, 1] + d * s[0, 2] - v[0, 1],
     s[0, 2] - v[0, 2],
     s[0, 3] + d * (s[0, 4] + d * v[0, 5]) - v[0, 3],
     s[0, 4] + d * v[0, 5] - v[0, 4],
     ],
    [s[1, 0] + d * (s[1, 1] + d * (s[1, 2] + d * (s[1, 3] + d * v[1, 4]))) - v[1, 0],
     s[1, 1] + d * (s[1, 2] + d * (s[1, 3] + d * (s[1, 4] + d * v[1, 5]))) - v[1, 1],
     s[1, 2] + d * (s[1, 3] + d * (s[1, 4] + d * v[1, 5])) - v[1, 2],
     s[1, 3] + d * (s[1, 4] + d * v[1, 5]) - v[1, 3],
     s[1, 4] + d * v[1, 5] - v[1, 4]
     ]
    ])
  np.testing.assert_allclose(actual, expected)


def test_n_step_returns(transitions):
  # given
  s = transitions.signals
  t = transitions.terminal
  v = transitions.values

  d = 0.9
  n_step = 4
  # when
  actual = discounted_nstep(s, v, t, discount_factor=d, n=n_step)
  # then
  expected = np.array([
    [s[0, 0] + d * (s[0, 1] + d * s[0, 2]),
     s[0, 1] + d * s[0, 2],
     s[0, 2],
     s[0, 3] + d * (s[0, 4] + d * v[0, 5]),
     s[0, 4] + d * v[0, 5],
     ],
    [s[1, 0] + d * (s[1, 1] + d * (s[1, 2] + d * (s[1, 3] + d * v[1, 4]))),
     s[1, 1] + d * (s[1, 2] + d * (s[1, 3] + d * (s[1, 4] + d * v[1, 5]))),
     s[1, 2] + d * (s[1, 3] + d * (s[1, 4] + d * v[1, 5])),
     s[1, 3] + d * (s[1, 4] + d * v[1, 5]),
     s[1, 4] + d * v[1, 5]
     ]
    ])
  np.testing.assert_allclose(actual, expected)


def test_ge_returns(transitions):
  # given

  s = transitions.signals
  t = transitions.terminal
  v = transitions.values

  d = 0.8
  steps = 0.9

  # when
  actual = discounted_ge(s, v, t, d, steps)
  # then
  expected = np.array([
    [(s[0, 0] + d * v[0, 1] - v[0, 0]) + d * steps * (
        (s[0, 1] + d * v[0, 2] - v[0, 1]) + d * steps * (s[0, 2] - v[0, 2])),
     (s[0, 1] + d * v[0, 2] - v[0, 1]) + d * steps * (s[0, 2] - v[0, 2]),
     (s[0, 2] - v[0, 2]),
     (s[0, 3] + d * v[0, 4] - v[0, 3]) + d * steps * (s[0, 4] + d * v[0, 5] - v[0, 4]),
     (s[0, 4] + d * v[0, 5] - v[0, 4]),
     ],
    [(s[1, 0] + d * v[1, 1] - v[1, 0]) + d * steps * ((s[1, 1] + d * v[1, 2] - v[1, 1]) + d * steps * (
        (s[1, 2] + d * v[1, 3] - v[1, 2]) + d * steps * (
        (s[1, 3] + d * v[1, 4] - v[1, 3]) + d * steps * (s[1, 4] + d * v[1, 5] - v[1, 4])))),
     (s[1, 1] + d * v[1, 2] - v[1, 1]) + d * steps * ((s[1, 2] + d * v[1, 3] - v[1, 2]) + d * steps * (
         (s[1, 3] + d * v[1, 4] - v[1, 3]) + d * steps * (s[1, 4] + d * v[1, 5] - v[1, 4]))),
     (s[1, 2] + d * v[1, 3] - v[1, 2]) + d * steps * ((s[1, 3] + d * v[1, 4] - v[1, 3]) + d * steps * (
         s[1, 4] + d * v[1, 5] - v[1, 4])),
     (s[1, 3] + d * v[1, 4] - v[1, 3]) + d * steps * (s[1, 4] + d * v[1, 5] - v[1, 4]),
     (s[1, 4] + d * v[1, 5] - v[1, 4])
     ]
    ]) + v[:, :-1]
  np.testing.assert_allclose(actual, expected, atol=1e-3)


if __name__ == '__main__':

  test_discounted_gae_returns(sample_transitions())
  test_ge_returns(sample_transitions())
  test_n_step_advantage_returns(sample_transitions())
  test_n_step_returns(sample_transitions())
