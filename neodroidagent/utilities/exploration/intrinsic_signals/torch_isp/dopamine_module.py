#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABCMeta

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""

__all__ = ["DopamineISP"]

from draugr.writers import Writer
from neodroid.utilities import EnvironmentSnapshot
from neodroidagent.utilities.exploration.intrinsic_signals.intrinsic_signal_provider import (
    IntrinsicSignalProvider,
)
from neodroidagent.utilities.exploration.intrinsic_signals.torch_isp.torch_isp_module import (
    TorchISPMeta,
)


class DopamineISP(IntrinsicSignalProvider, TorchISPMeta):
    def sample(
        self,
        environment_snapshot: EnvironmentSnapshot,
        *,
        writer: Writer = None,
        **kwargs
    ):
        return 0
