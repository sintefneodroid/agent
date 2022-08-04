#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Christian Heider Nielsen"
__doc__ = ""
__all__ = ["ExplorationSpecification"]


from collections import namedtuple

ExplorationSpecification = namedtuple(
    "ExplorationSpecification", ["start", "end", "decay"]
)
