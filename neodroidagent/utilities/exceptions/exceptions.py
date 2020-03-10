#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

from draugr.python_utilities.exceptions import NoData

__all__ = [
    "NoData",
    "NoAgent",
    "NoProcedure",
    "NoTrajectoryException",
    "ActionSpaceNotSupported",
]


class NoProcedure(Exception):
    """

"""

    def __init__(self, msg="No Procedure"):
        Exception.__init__(self, msg)


class NoAgent(Exception):
    """

"""

    def __init__(self, msg="No Agent"):
        Exception.__init__(self, msg)


class NoTrajectoryException(Exception):
    """

"""

    def __init__(self, msg="No Trajectory Available"):
        Exception.__init__(self, msg)


class ActionSpaceNotSupported(Exception):
    """

"""

    def __init__(self, msg="Action space not supported by agent"):
        Exception.__init__(self, msg)
