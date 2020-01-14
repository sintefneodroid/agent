# !/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Christian Heider Nielsen"
__all__ = ["TrajectoryPoint", "AdvantageMemoryPoint"]


from collections import namedtuple

TrajectoryPoint = namedtuple("TrajectoryTrace", ("signal", "log_prob", "entropy"))

AdvantageMemoryPoint = namedtuple(
    "AdvantageMemory",
    (
        "state",
        "action",
        "action_prob",
        "value_estimate",
        "advantage",
        "discounted_return",
    ),
)
