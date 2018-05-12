import math

import pylab as plt

plt.ion()

expand_x = 0
expand_y = 1

inc_x = 2
dec_x = 3

inc_y = 4
dec_y = 5


def snake_space_filling_generator():
  x = 0
  y = 0
  state = expand_x
  yield x, y

  while 1:
    if state is expand_x:
      x += 1
      state = inc_y
    elif state is inc_x:
      x += 1
      if y is x:
        state = dec_y
    elif state is dec_x:
      x -= 1
      if x is 0:
        state = expand_y

    elif state is expand_y:
      y += 1
      state = inc_x
    elif state is inc_y:
      y += 1
      if y is x:
        state = dec_x
    elif state is dec_y:
      y -= 1
      if y is 0:
        state = expand_x

    yield x, y


if __name__ == '__main__':
  num = 300000
  annotate = False
  scaling_factor = 0.1

  generator = snake_space_filling_generator()
  points = [(x, y) for ((x, y), i) in zip(generator, range(num)) if i < num]

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
