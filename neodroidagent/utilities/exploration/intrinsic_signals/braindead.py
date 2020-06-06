#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25/04/2020
           """

__all__ = ["BraindeadIntrinsicSignalProvider"]

from abc import abstractmethod
from typing import Sequence

from draugr.writers import Writer
from neodroid.utilities import (
    ActionSpace,
    EnvironmentSnapshot,
    ObservationSpace,
    SignalSpace,
)
from neodroidagent.utilities import IntrinsicSignalProvider
from warg import drop_unused_kws


class BraindeadIntrinsicSignalProvider(IntrinsicSignalProvider):
    def sample(
        self,
        environment_snapshot: EnvironmentSnapshot,
        *,
        writer: Writer = None,
        **kwargs
    ) -> Sequence:
        """

    @param environment_snapshot:
    @type environment_snapshot:
    @param writer:
    @type writer:
    @param kwargs:
    @type kwargs:
    @return:
    @rtype:
    """
        return self._signal_space.n * [0]
