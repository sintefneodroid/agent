# !/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Christian Heider Nielsen"
__all__ = ["SampleTrajectoryPoint", "SamplePoint"]

from collections import namedtuple

SampleTrajectoryPoint = namedtuple(
    "SampleTrajectoryPoint", ("signal", "terminated", "action", "distribution")
)

SamplePoint = namedtuple("SamplePoint", ("action", "distribution"))
