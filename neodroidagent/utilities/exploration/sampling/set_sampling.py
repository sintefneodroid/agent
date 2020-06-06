#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any

__author__ = "Christian Heider Nielsen"

import numpy

__all__ = ["sample"]


def sample(iter_set: iter) -> Any:
    """

    @param iter_set:
    @type iter_set:
    @return:
    @rtype:
    """
    a = list(iter_set)
    if len(a):
        idx = numpy.random.randint(0, len(a))
        return a[idx]
