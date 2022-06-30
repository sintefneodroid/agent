#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

import math
from enum import Enum

from sorcery import assigned_names

__all__ = ["snake_space_filling_generator"]


class States(Enum):
    (expand_x, expand_y, inc_x, dec_x, inc_y, dec_y) = assigned_names()


def snake_space_filling_generator():
    """ """
    x = 0
    y = 0
    state = States.expand_x
    yield x, y

    while True:
        if state == States.expand_x:
            x += 1
            state = States.inc_y
        elif state == States.inc_x:
            x += 1
            if y == x:
                state = States.dec_y
        elif state == States.dec_x:
            x -= 1
            if x == 0:
                state = States.expand_y

        elif state == States.expand_y:
            y += 1
            state = States.inc_x
        elif state == States.inc_y:
            y += 1
            if y == x:
                state = States.dec_x
        elif state == States.dec_y:
            y -= 1
            if y == 0:
                state = States.expand_x

        yield x, y


if __name__ == "__main__":

    def ijhasd():
        from matplotlib import pyplot

        # pyplot.ion()
        num = 100
        annotate = False
        scaling_factor = 0.1

        generator = snake_space_filling_generator()
        points = [(x, y) for (i, (x, y)) in zip(range(num), generator)]
        outsider_point = generator.__next__()
        # ------ Plotting ------
        xs, ys = zip(*points)

        end = math.sqrt(num)
        end_scaled = end * scaling_factor
        if end_scaled < 4:
            end_scaled = 4
        size = (end_scaled, end_scaled)
        fig, ax = pyplot.subplots(figsize=size)
        line = pyplot.Line2D(xs, ys)
        ax.add_line(line)

        if annotate:
            ax.scatter(xs, ys, 160)
            for i, txt in enumerate(range(num)):
                ax.annotate(
                    txt,
                    (xs[i], ys[i]),
                    fontsize=8,
                    color="black",
                    va="center",
                    ha="center",
                )

        ax.axis((-1, end + 1, -1, end + 1))

        pyplot.show()

    ijhasd()
