#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 19/01/2020
           """

from typing import Any, Tuple

import numpy

from draugr.writers import MockWriter, Writer
from neodroid.utilities import (
    ActionSpace,
    EnvironmentSnapshot,
    ObservationSpace,
    SignalSpace,
)
from neodroidagent.agents import Agent


class LinearFeatureBaselineAgent(Agent):
    """

    """

    def __init__(self, reg_coeff=1e-5, **kwargs):
        super().__init__(**kwargs)
        self._coeffs = None
        self._reg_coeff = reg_coeff

    def __build__(
        self,
        *,
        observation_space: ObservationSpace = None,
        action_space: ActionSpace = None,
        signal_space: SignalSpace = None,
        **kwargs
    ) -> None:
        pass

    def eval(self) -> None:
        """

        """
        pass

    def load(self, *, save_directory, **kwargs) -> None:
        """

        @param save_directory:
        @type save_directory:
        @param kwargs:
        @type kwargs:
        """
        pass

    def save(self, *, save_directory, **kwargs) -> None:
        """

        @param save_directory:
        @type save_directory:
        @param kwargs:
        @type kwargs:
        @return:
        @rtype:
        """
        return self._coeffs

    def _remember(self, **kwargs) -> None:
        pass

    def _sample(
        self,
        state: EnvironmentSnapshot,
        *args,
        deterministic: bool = False,
        metric_writer: Writer = MockWriter(),
        **kwargs
    ) -> Tuple[Any, ...]:
        if self._coeffs is None:
            return numpy.zeros(len(state.signal))

        return self.extract_features(state).dot(self._coeffs)

    def extract_features(self, state: EnvironmentSnapshot) -> Any:
        """Feature extraction"""
        obs = numpy.clip(state.observables, -10, 10)
        length = len(state.signal)
        al = numpy.arange(length).reshape(-1, 1) / 100.0
        sad = numpy.concatenate(
            [obs, obs ** 2, al, al ** 2, al ** 3, numpy.ones((length, 1))], axis=1
        )
        return sad

    def _update(
        self, states, *args, metric_writer: Writer = MockWriter(), **kwargs
    ) -> Any:
        featmat = numpy.concatenate([self.extract_features(path) for path in states])
        returns = numpy.concatenate([path.signal for path in states])
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = numpy.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * numpy.identity(featmat.shape[1]),
                featmat.T.dot(returns),
                rcond=-1,
            )[0]
            if not numpy.any(numpy.isnan(self._coeffs)):
                break
            reg_coeff *= 10


if __name__ == "__main__":
    agent = LinearFeatureBaselineAgent()
