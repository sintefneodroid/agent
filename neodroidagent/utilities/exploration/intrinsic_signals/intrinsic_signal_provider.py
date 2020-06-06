#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25/04/2020
           """

__all__ = ["IntrinsicSignalProvider"]

from abc import abstractmethod
from typing import Sequence

from draugr.writers import Writer
from neodroid.utilities import (
    ActionSpace,
    EnvironmentSnapshot,
    ObservationSpace,
    SignalSpace,
)
from warg import drop_unused_kws


class IntrinsicSignalProvider:
    """
  A callable module that congests observations and provide augmented signals external to the
  environment/MDP provided objective signals for
  the
  learning
  control model
  """

    @drop_unused_kws
    def __init__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        signal_space: SignalSpace,
    ):
        """

    @param observation_space:
    @type observation_space:
    @param action_space:
    @type action_space:
    @param signal_space:
    @type signal_space:
    """
        self._observation_space = observation_space
        self._action_space = action_space
        self._signal_space = signal_space

    def __call__(self, environment_snapshot: EnvironmentSnapshot) -> Sequence:
        """

    @param environment_snapshot:
    @type environment_snapshot:
    @return:
    @rtype:
    """
        return self.sample(environment_snapshot)

    @abstractmethod
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
    """
        raise NotImplemented
