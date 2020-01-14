#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import namedtuple

__author__ = "Christian Heider Nielsen"
__doc__ = ""
__all__ = ["ExplorationSpecification"]

ExplorationSpecification = namedtuple(
    "ExplorationSpecification", ["start", "end", "decay"]
)
