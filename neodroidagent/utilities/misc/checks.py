#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11/02/2020
           """

from typing import Any, Sequence
from warnings import warn

__all__ = ["check_tensorised_shapes"]


def check_tensorised_shapes(tensorised: Sequence) -> None:
    aa = iter(tensorised)
    a = next(aa).shape[:-1]
    try:
        while True:
            b = next(aa).shape[:-1]
            if a != b:
                warn(f"{a},{b}, Does not match!")
            a = b
    except StopIteration:
        pass
