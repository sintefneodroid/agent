#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABCMeta

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""

__all__ = ["TorchISPMeta", "TorchISPModule"]

from draugr.writers import Writer
from neodroid.utilities import EnvironmentSnapshot
from neodroidagent.utilities.exploration.intrinsic_signals.intrinsic_signal_provider import (
    IntrinsicSignalProvider,
)


class TorchISPMeta(metaclass=ABCMeta):
    pass


class TorchISPModule(IntrinsicSignalProvider, TorchISPMeta):
    def sample(
        self,
        environment_snapshot: EnvironmentSnapshot,
        *,
        writer: Writer = None,
        **kwargs
    ):
        raise NotImplemented
