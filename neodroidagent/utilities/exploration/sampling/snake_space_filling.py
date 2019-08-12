#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
import math
from enum import Enum, auto

import matplotlib.pyplot as plt

plt.ion()


class States(Enum):
  expand_x = auto()
  expand_y = auto()

  inc_x = auto()
  dec_x = auto()

  inc_y = auto()
  dec_y = auto()


def snake_space_filling_generator():
  x = 0
  y = 0
  state = States.expand_x
  yield x, y

  while True:
    if state is States.expand_x:
      x += 1
      state = States.inc_y
    elif state is States.inc_x:
      x += 1
      if y is x:
        state = States.dec_y
    elif state is States.dec_x:
      x -= 1
      if x is 0:
        state = States.expand_y

    elif state is States.expand_y:
      y += 1
      state = States.inc_x
    elif state is States.inc_y:
      y += 1
      if y is x:
        state = States.dec_x
    elif state is States.dec_y:
      y -= 1
      if y is 0:
        state = States.expand_x

    yield x, y


if __name__ == '__main__':
  num = 10000
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
  fig, ax = plt.subplots(figsize=size)
  line = plt.Line2D(xs, ys)
  ax.add_line(line)

  if annotate:
    ax.scatter(xs, ys, 160)
    for i, txt in enumerate(range(num)):
      ax.annotate(txt, (xs[i], ys[i]), fontsize=8, color='black', va='center', ha='center')

  ax.axis((-1, end + 1, -1, end + 1))

  plt.show()
