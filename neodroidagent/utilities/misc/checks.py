#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11/02/2020
           """

from warnings import warn


def check_tensorised_shapes(tensorised) -> None:

    return

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
