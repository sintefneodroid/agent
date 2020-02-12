#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 19/01/2020
           """

from typing import Any, Tuple

import numpy as np

from draugr import Writer, MockWriter
from neodroid.utilities import (
    EnvironmentSnapshot,
    ObservationSpace,
    ActionSpace,
    SignalSpace,
)
from neodroidagent.agents import Agent


class LinearFeatureBaselineAgent(Agent):
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
        pass

    def load(self, *, save_directory, **kwargs) -> None:
        pass

    def save(self, *, save_directory, **kwargs) -> None:
        return self._coeffs

    def _remember(self, **kwargs):
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
            return np.zeros(len(state.signal))

        return self.extract_features(state).dot(self._coeffs)

    def extract_features(self, state: EnvironmentSnapshot) -> Any:
        """Feature extraction"""
        obs = np.clip(state.observables, -10, 10)
        length = len(state.signal)
        al = np.arange(length).reshape(-1, 1) / 100.0
        sad = np.concatenate(
            [obs, obs ** 2, al, al ** 2, al ** 3, np.ones((length, 1))], axis=1
        )
        return sad

    def _update(
        self, states, *args, metric_writer: Writer = MockWriter(), **kwargs
    ) -> Any:
        featmat = np.concatenate([self.extract_features(path) for path in states])
        returns = np.concatenate([path.signal for path in states])
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns),
                rcond=-1,
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10


if __name__ == "__main__":
    agent = LinearFeatureBaselineAgent()
