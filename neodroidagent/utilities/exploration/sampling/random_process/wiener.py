#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .random_process import RandomProcess

__author__ = "Christian Heider Nielsen"

from math import sqrt

from matplotlib import pyplot
import numpy
from scipy.stats import norm

__all__ = ["WienerProcess", "wiener"]


class WienerProcess(RandomProcess):
    def reset(self):
        pass

    def __init__(self, delta, dt, initial, size=1, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
        self.dt = dt
        self.last_x = initial

    def sample(self, size=1):
        x = self.last_x + norm.rvs(scale=self.delta**2 * self.dt)
        self.last_x = x
        return x


def wiener(x0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

    X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.

    Written as an iteration scheme,

    X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
    using numpy.asarray(x0)).
    The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
    The number of steps to take.
    dt : float
    The time step.
    delta : float
    delta determines the 'speed' of the Brownian motion.  The random variable
    of the position at time t, X(t), has a normal distribution whose mean is
    the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
    If `out` is not None, it specifies the array in which to put the
    result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array."""

    x0 = numpy.asarray(x0)

    r = norm.rvs(
        size=x0.shape + (n,), scale=delta * sqrt(dt)
    )  # For each element of x0, generate a sample of n numbers from a normal distribution.

    # If `out` was not given, create an output array.
    if out is None:
        out = numpy.empty(r.shape)

    numpy.cumsum(
        r, axis=-1, out=out
    )  # This computes the Brownian motion by forming the cumulative sum of the random samples.

    out += numpy.expand_dims(x0, axis=-1)  # Add the initial condition.

    return out


if __name__ == "__main__":

    def main_1d():
        # The Wiener process parameter.
        delta = 0.1
        # Total time.
        T = 1
        # Number of steps.
        N = 50
        # Time step size
        dt = T / N
        # Number of realizations to generate.
        m = 5
        # Create an empty array to store the realizations.
        x = numpy.empty((m, N + 1))
        # Initial values of x.
        x[:, 0] = 0

        wiener(x[:, 0], N, dt, delta, out=x[:, 1:])

        t = numpy.linspace(0.0, N * dt, N + 1)
        plot_1d_trajectory(x, t, m)

    def main_2d():
        # The Wiener process parameter.
        delta = 0.25
        # Total time.
        T = 1.0
        # Number of steps.
        N = 50
        # Time step size
        dt = T / N
        # Initial values of x.
        x = numpy.empty((2, N + 1))
        x[:, 0] = 0.0

        wiener(x[:, 0], N, dt, delta, out=x[:, 1:])

        plot_2d_trajectory(x)

    def main_3d():
        # The Wiener process parameter.
        delta = 0.25
        # Total time.
        T = 1.0
        # Number of steps.
        N = 1000
        # Time step size
        dt = T / N
        # Initial values of x.
        x = numpy.zeros((3, N + 1))

        wiener(x[:, 0], N, dt, delta, out=x[:, 1:])

        plot_3d_trajectory(x)

    def plot_3d_trajectory(x):
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x[0], x[1], x[2])

        ax.scatter(x[0, 0], x[1, 0], x[2, 0], "go")
        ax.scatter(x[0, -1], x[1, -1], x[2, -1], "ro")

        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")

        pyplot.title("3D Brownian Motion")
        pyplot.axis("equal")
        pyplot.show()

    def plot_1d_trajectory(x, t, m):
        for k in range(m):
            pyplot.plot(t, x[k])
        pyplot.xlabel("t", fontsize=16)
        pyplot.ylabel("x", fontsize=16)
        pyplot.grid(True)
        pyplot.show()

    def plot_2d_trajectory(x):
        # Plot the 2D trajectory.
        pyplot.plot(x[0], x[1])

        # Mark the start and end points.
        pyplot.plot(x[0, 0], x[1, 0], "go")
        pyplot.plot(x[0, -1], x[1, -1], "ro")

        # More plot decorations.
        pyplot.title("2D Brownian Motion")
        pyplot.xlabel("x", fontsize=16)
        pyplot.ylabel("y", fontsize=16)
        pyplot.axis("equal")
        pyplot.grid(True)
        pyplot.show()

    def main_class():
        # The Wiener process parameter.
        delta = 1
        # Total time.
        T = 1.0
        # Number of steps.
        N = 50
        # Time step size
        dt = T / N
        # Initial values of x.
        x = numpy.empty((2, N + 1))
        x[:, 0] = 0.0

        brownian = WienerProcess(delta, dt, 0)

        for i in range(N):
            print(brownian.sample())

    main_3d()
