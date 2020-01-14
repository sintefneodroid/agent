#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sequence
import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """
__all__ = ["mini_batch_iter"]

from neodroidagent.common.transitions import ValuedTransition


def mini_batch_iter(mini_batch_size: int, batch: Sequence[ValuedTransition]) -> iter:
    batch_size = len(batch)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = numpy.random.randint(0, batch_size, mini_batch_size)
        a = batch[:, rand_ids]
        yield ValuedTransition(*a)
