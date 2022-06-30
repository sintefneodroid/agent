#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 19/01/2020
           """

from typing import Any, Optional

import numpy
from draugr.writers import MockWriter, Writer

from neodroid.utilities import (
    EnvironmentSnapshot,
)


class LinearFeatureBaseline:
    """
        Linear (polynomial) time-state dependent return baseline model
    (see. Duan et al. 2016, "Benchmarking Deep Reinforcement Learning for Continuous Control", ICML)

    Predicts value of GAE (Generalized Advantage Estimation) Baseline.
    """

    def __init__(self, reg_coeff: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self._linear_coefficients = None
        self._l2_reg_coefficient = reg_coeff

    def sample(
        self,
        state: EnvironmentSnapshot,
        *args,
        metric_writer: Optional[Writer] = MockWriter(),
        **kwargs
    ) -> Any:
        """

        samples signal

        :param state:
        :type state:
        :param args:
        :type args:
        :param metric_writer:
        :type metric_writer:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if self._linear_coefficients is None:
            return numpy.zeros(len(state["rewards"]))
        return self.extract_features(state).dot(self._linear_coefficients)

    def extract_features(self, state: EnvironmentSnapshot) -> Any:
        """

        b0 + b1*obs + b2*obs^2 + b3*t + b4*t^2+  b5*t^3


        :param state:
        :type state:
        :return:
        :rtype:
        """
        obs = numpy.clip(state["observations"], -10, 10)
        trajectory_length = len(state["rewards"])
        # obs = numpy.clip(state.observables, -10, 10) # TODO: Clipping
        # obs = state.observables
        time_steps = numpy.arange(trajectory_length).reshape(-1, 1) / 100.0
        return numpy.concatenate(
            [
                obs,
                obs**2,
                time_steps,
                time_steps**2,
                time_steps**3,
                numpy.ones((trajectory_length, 1)),
            ],
            axis=1,
        )

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def update(
        self,
        trajectories,
        *args,
        metric_writer: Optional[Writer] = MockWriter(),
        attempts=5,
        **kwargs
    ) -> Any:
        """

        Fit the linear baseline model (signal estimator) with the provided paths via damped least squares


        :param trajectories:
        :type trajectories:
        :param args:
        :type args:
        :param metric_writer:
        :type metric_writer:
        :param attempts:
        :type attempts:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        features_matrix = numpy.concatenate(
            [self.extract_features(trajectory) for trajectory in trajectories]
        )
        returns_matrix = numpy.concatenate(
            [trajectory["returns"] for trajectory in trajectories]
        )
        # returns_matrix = numpy.concatenate([path.returns for path in states])
        c_regularisation_coeff = self._l2_reg_coefficient
        id_fm = numpy.identity(features_matrix.shape[1])
        for _ in range(attempts):
            self._linear_coefficients = numpy.linalg.lstsq(
                features_matrix.T.dot(features_matrix) + c_regularisation_coeff * id_fm,
                features_matrix.T.dot(returns_matrix),
                rcond=-1,
            )[0]
            if not numpy.any(numpy.isnan(self._linear_coefficients)):
                break  # Non-Nan solution found
            c_regularisation_coeff *= 10


if __name__ == "__main__":
    agent = LinearFeatureBaseline()
