#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27/02/2020
           """

import pytest

from warg import NOD

ATOL = 1e-3


def sample_transitions():
    signals = numpy.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], numpy.float32)
    terminals = numpy.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0]], numpy.float32)
    values = numpy.array(
        [[-100, 10, 20, 30, 40, 50], [-150, 15, 25, 35, 45, 55]], numpy.float32
    )  # Future values

    return NOD({"signals": signals, "terminals": terminals, "values": values})


@pytest.fixture
def transitions():
    return sample_transitions()


import numpy

from neodroidagent.utilities.signal.experimental.generalised_advantage import (
    discounted_gae,
    discounted_ge,
)
from neodroidagent.utilities.signal.experimental.nstep import (
    discounted_nstep,
    discounted_nstep_adv,
)


def test_discounted_gae_returns(transitions, steps=0.9, d=0.8):
    # given
    s = transitions.signals
    t = transitions.terminals
    v = transitions.values

    # when
    actual = discounted_gae(
        signals=s, values=v, terminals=t, discount_factor=d, step_factor=steps
    )
    # then
    expected = numpy.array(
        [
            [
                (s[0, 0] + d * v[0, 1] - v[0, 0])
                + d
                * steps
                * ((s[0, 1] + d * v[0, 2] - v[0, 1]) + d * steps * (s[0, 2] - v[0, 2])),
                (s[0, 1] + d * v[0, 2] - v[0, 1]) + d * steps * (s[0, 2] - v[0, 2]),
                (s[0, 2] - v[0, 2]),
                (s[0, 3] + d * v[0, 4] - v[0, 3])
                + d * steps * (s[0, 4] + d * v[0, 5] - v[0, 4]),
                (s[0, 4] + d * v[0, 5] - v[0, 4]),
            ],
            [
                (s[1, 0] + d * v[1, 1] - v[1, 0])
                + d
                * steps
                * (
                    (s[1, 1] + d * v[1, 2] - v[1, 1])
                    + d
                    * steps
                    * (
                        (s[1, 2] + d * v[1, 3] - v[1, 2])
                        + d
                        * steps
                        * (
                            (s[1, 3] + d * v[1, 4] - v[1, 3])
                            + d * steps * (s[1, 4] + d * v[1, 5] - v[1, 4])
                        )
                    )
                ),
                (s[1, 1] + d * v[1, 2] - v[1, 1])
                + d
                * steps
                * (
                    (s[1, 2] + d * v[1, 3] - v[1, 2])
                    + d
                    * steps
                    * (
                        (s[1, 3] + d * v[1, 4] - v[1, 3])
                        + d * steps * (s[1, 4] + d * v[1, 5] - v[1, 4])
                    )
                ),
                (s[1, 2] + d * v[1, 3] - v[1, 2])
                + d
                * steps
                * (
                    (s[1, 3] + d * v[1, 4] - v[1, 3])
                    + d * steps * (s[1, 4] + d * v[1, 5] - v[1, 4])
                ),
                (s[1, 3] + d * v[1, 4] - v[1, 3])
                + d * steps * (s[1, 4] + d * v[1, 5] - v[1, 4]),
                (s[1, 4] + d * v[1, 5] - v[1, 4]),
            ],
        ]
    )
    numpy.testing.assert_allclose(expected, actual, atol=ATOL)


def test_n_step_advantage_returns(transitions, d=0.9, n_step=4):
    # given
    s = transitions.signals
    t = transitions.terminals
    v = transitions.values

    # when
    actual = discounted_nstep_adv(s, v, t, discount_factor=d, n=n_step)
    # then
    expected = numpy.array(
        [
            [
                s[0, 0] + d * (s[0, 1] + d * s[0, 2]) - v[0, 0],
                s[0, 1] + d * s[0, 2] - v[0, 1],
                s[0, 2] - v[0, 2],
                s[0, 3] + d * (s[0, 4] + d * v[0, 5]) - v[0, 3],
                s[0, 4] + d * v[0, 5] - v[0, 4],
            ],
            [
                s[1, 0]
                + d * (s[1, 1] + d * (s[1, 2] + d * (s[1, 3] + d * v[1, 4])))
                - v[1, 0],
                s[1, 1]
                + d * (s[1, 2] + d * (s[1, 3] + d * (s[1, 4] + d * v[1, 5])))
                - v[1, 1],
                s[1, 2] + d * (s[1, 3] + d * (s[1, 4] + d * v[1, 5])) - v[1, 2],
                s[1, 3] + d * (s[1, 4] + d * v[1, 5]) - v[1, 3],
                s[1, 4] + d * v[1, 5] - v[1, 4],
            ],
        ]
    )
    numpy.testing.assert_allclose(actual, expected, atol=ATOL)


def test_n_step_returns(transitions, d=0.9, n_step=4):
    # given
    s = transitions.signals
    t = transitions.terminals
    v = transitions.values

    # when
    actual = discounted_nstep(s, v, t, discount_factor=d, n=n_step)
    # then
    expected = numpy.array(
        [
            [
                s[0, 0] + d * (s[0, 1] + d * s[0, 2]),
                s[0, 1] + d * s[0, 2],
                s[0, 2],
                s[0, 3] + d * (s[0, 4] + d * v[0, 5]),
                s[0, 4] + d * v[0, 5],
            ],
            [
                s[1, 0] + d * (s[1, 1] + d * (s[1, 2] + d * (s[1, 3] + d * v[1, 4]))),
                s[1, 1] + d * (s[1, 2] + d * (s[1, 3] + d * (s[1, 4] + d * v[1, 5]))),
                s[1, 2] + d * (s[1, 3] + d * (s[1, 4] + d * v[1, 5])),
                s[1, 3] + d * (s[1, 4] + d * v[1, 5]),
                s[1, 4] + d * v[1, 5],
            ],
        ]
    )
    numpy.testing.assert_allclose(actual, expected, atol=ATOL)


def test_ge_returns(transitions, d=0.8, steps=0.9):
    # given

    s = transitions.signals
    t = transitions.terminals
    v = transitions.values

    # when
    actual = discounted_ge(s, v, t, d, steps)
    # then
    expected = (
        numpy.array(
            [
                [
                    (s[0, 0] + d * v[0, 1] - v[0, 0])
                    + d
                    * steps
                    * (
                        (s[0, 1] + d * v[0, 2] - v[0, 1])
                        + d * steps * (s[0, 2] - v[0, 2])
                    ),
                    (s[0, 1] + d * v[0, 2] - v[0, 1]) + d * steps * (s[0, 2] - v[0, 2]),
                    (s[0, 2] - v[0, 2]),
                    (s[0, 3] + d * v[0, 4] - v[0, 3])
                    + d * steps * (s[0, 4] + d * v[0, 5] - v[0, 4]),
                    (s[0, 4] + d * v[0, 5] - v[0, 4]),
                ],
                [
                    (s[1, 0] + d * v[1, 1] - v[1, 0])
                    + d
                    * steps
                    * (
                        (s[1, 1] + d * v[1, 2] - v[1, 1])
                        + d
                        * steps
                        * (
                            (s[1, 2] + d * v[1, 3] - v[1, 2])
                            + d
                            * steps
                            * (
                                (s[1, 3] + d * v[1, 4] - v[1, 3])
                                + d * steps * (s[1, 4] + d * v[1, 5] - v[1, 4])
                            )
                        )
                    ),
                    (s[1, 1] + d * v[1, 2] - v[1, 1])
                    + d
                    * steps
                    * (
                        (s[1, 2] + d * v[1, 3] - v[1, 2])
                        + d
                        * steps
                        * (
                            (s[1, 3] + d * v[1, 4] - v[1, 3])
                            + d * steps * (s[1, 4] + d * v[1, 5] - v[1, 4])
                        )
                    ),
                    (s[1, 2] + d * v[1, 3] - v[1, 2])
                    + d
                    * steps
                    * (
                        (s[1, 3] + d * v[1, 4] - v[1, 3])
                        + d * steps * (s[1, 4] + d * v[1, 5] - v[1, 4])
                    ),
                    (s[1, 3] + d * v[1, 4] - v[1, 3])
                    + d * steps * (s[1, 4] + d * v[1, 5] - v[1, 4]),
                    (s[1, 4] + d * v[1, 5] - v[1, 4]),
                ],
            ]
        )
        + v[:, :-1]
    )
    numpy.testing.assert_allclose(actual, expected, atol=ATOL)


if __name__ == "__main__":
    test_discounted_gae_returns(sample_transitions())
    test_ge_returns(sample_transitions())
    test_n_step_advantage_returns(sample_transitions())
    test_n_step_returns(sample_transitions())
