#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest


from warg import NOD

__author__ = "Christian Heider Nielsen"
__doc__ = ""

import numpy


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
