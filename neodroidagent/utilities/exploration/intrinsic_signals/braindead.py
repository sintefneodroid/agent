#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25/04/2020
           """

__all__ = ["BraindeadIntrinsicSignalProvider"]

from typing import Sequence

from draugr.writers import Writer

from neodroid.utilities import (
    EnvironmentSnapshot,
)
from neodroidagent.utilities.exploration.intrinsic_signals.intrinsic_signal_provider import (
    IntrinsicSignalProvider,
)


class BraindeadIntrinsicSignalProvider(IntrinsicSignalProvider):
    def sample(
        self,
        environment_snapshot: EnvironmentSnapshot,
        *,
        writer: Writer = None,
        **kwargs
    ) -> Sequence:
        """

        :param environment_snapshot:
        :type environment_snapshot:
        :param writer:
        :type writer:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return self._signal_space.n * [0]
