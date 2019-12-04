#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any

__author__ = "Christian Heider Nielsen"

import numpy


def sample(iter_set: iter) -> Any:
    a = list(iter_set)
    if len(a):
        idx = numpy.random.randint(0, len(a))
        return a[idx]
