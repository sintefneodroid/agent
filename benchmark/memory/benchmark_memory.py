#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy

from neodroidagent.utilities import TrajectoryBuffer, TransitionBuffer

__author__ = "Christian Heider Nielsen"
__doc__ = ""


def benchmark_transition_buffer_list():
    rb = TransitionBuffer()
    a = numpy.random.random((999, 999))
    for e in a:
        rb.add_transition_point(e, None, None, None, None)
    b, *_ = rb.sample_transition_points(999)


def benchmark_trajectory_list():
    rb = TrajectoryBuffer()
    a = numpy.random.random((999, 999))
    for e in a:
        rb.add_trajectory_point(e, None, None)
    b, *_ = rb.retrieve_trajectory()


if __name__ == "__main__":
    benchmark_transition_buffer_list()
    benchmark_trajectory_list()
