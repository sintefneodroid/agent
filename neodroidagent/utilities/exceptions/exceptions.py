#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"


class HasNoEnvError(Exception):
    """
Raised when an agent has no environment assigned and some implicit next or step called.
"""

    def __init__(self, msg="Agent has no env assigned"):
        Exception.__init__(self, msg)


class NoEnvironment(Exception):
    """

"""

    def __init__(self, msg="No Environment"):
        Exception.__init__(self, msg)


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


class NoData(Exception):
    """

"""

    def __init__(self, msg="No Data Available"):
        Exception.__init__(self, msg)


class ActionSpaceNotSupported(Exception):
    """

"""

    def __init__(self, msg="Action space not supported by agent"):
        Exception.__init__(self, msg)
