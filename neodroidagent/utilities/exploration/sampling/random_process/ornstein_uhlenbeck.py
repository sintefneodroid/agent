#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .annealed_guassian import AnnealedGaussianProcess

__author__ = "Christian Heider Nielsen"

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
import numpy


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(
        self,
        *,
        theta: float = 0.15,
        mean: float = 0.0,
        sigma: float = 1.0,
        dt: float = 1e-2,
        x_0=None,
        sigma_min: float = None,
        n_steps_annealing: int = 1000,
        **kwargs
    ):
        super().__init__(
            mean=mean,
            sigma=sigma,
            sigma_min=sigma_min,
            n_steps_annealing=n_steps_annealing,
            **kwargs
        )
        self.theta = theta
        self.mean = mean
        self.dt = dt
        self.x_0 = x_0
        self.reset()

    def sample(self, size):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.current_sigma * numpy.sqrt(self.dt) * numpy.random.normal(size=size)
        )
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset(self):
        super().reset()
        self.x_prev = self.x_0 if self.x_0 is not None else numpy.zeros_like(self.x_0)


if __name__ == "__main__":

    random_process = OrnsteinUhlenbeckProcess(theta=0.5)

    for i in range(1000):
        print(random_process.sample((2, 1)))
