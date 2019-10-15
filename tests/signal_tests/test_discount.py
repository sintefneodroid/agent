#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroidagent.utilities.signal.experimental.discounting import valued_discount

__author__ = 'Christian Heider Nielsen'
__doc__ = ''

import numpy


def test_discounting_respects_episode_end():
  # given
  s = numpy.array([[1, 2, 3, 4, 5],
                   [5, 4, 3, 2, 1]
                   ],
                  numpy.float32)
  t = numpy.array([[0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0]
                   ],
                  numpy.float32)
  last = numpy.array([6, 0],
                     numpy.float32)
  d = 0.9
  # when
  actual = valued_discount(s, last, t, d)
  # then
  expected = numpy.array([
    [s[0, 0] + d * (s[0, 1] + d * s[0, 2]),
     s[0, 1] + d * s[0, 2],
     s[0, 2],
     s[0, 3] + d * (s[0, 4] + d * last[0]),
     s[0, 4] + d * last[0],
     ],
    [s[1, 0] + d * (s[1, 1] + d * (s[1, 2] + d * (s[1, 3] + d * (s[1, 4] + d * last[1])))),
     s[1, 1] + d * (s[1, 2] + d * (s[1, 3] + d * (s[1, 4] + d * last[1]))),
     s[1, 2] + d * (s[1, 3] + d * (s[1, 4] + d * last[1])),
     s[1, 3] + d * (s[1, 4] + d * last[1]),
     s[1, 4] + d * last[1]
     ]
    ])
  numpy.testing.assert_allclose(actual, expected)


def test_discounting_respects_episode_end_T():
  # given
  s = numpy.array([[1, 5],
                   [2, 4],
                   [3, 3],
                   [4, 2],
                   [5, 1]
                   ],
                  numpy.float32).T
  t = numpy.array([[0, 0],
                   [0, 0],
                   [1, 0],
                   [0, 0],
                   [0, 0]
                   ],
                  numpy.float32).T
  last = numpy.array([6, 0],
                     numpy.float32)
  d = 0.9
  # when
  actual = valued_discount(s, last, t, d)
  # then
  expected = numpy.array([
    [s[0, 0] + d * (s[0, 1] + d * s[0, 2]),
     s[0, 1] + d * s[0, 2],
     s[0, 2],
     s[0, 3] + d * (s[0, 4] + d * last[0]),
     s[0, 4] + d * last[0],
     ],
    [s[1, 0] + d * (s[1, 1] + d * (s[1, 2] + d * (s[1, 3] + d * (s[1, 4] + d * last[1])))),
     s[1, 1] + d * (s[1, 2] + d * (s[1, 3] + d * (s[1, 4] + d * last[1]))),
     s[1, 2] + d * (s[1, 3] + d * (s[1, 4] + d * last[1])),
     s[1, 3] + d * (s[1, 4] + d * last[1]),
     s[1, 4] + d * last[1]
     ]
    ])
  numpy.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
  test_discounting_respects_episode_end()
  test_discounting_respects_episode_end_T()
