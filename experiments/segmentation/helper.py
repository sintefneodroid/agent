import itertools
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np


def plot_img_array(img_array, n_col=3):
  n_row = len(img_array) // n_col

  f, plots = plt.subplots(n_row, n_col, sharex='all', sharey='all', figsize=(n_col * 4, n_row * 4))

  for i in range(len(img_array)):
    plots[i // n_col, i % n_col].imshow(img_array[i])


def plot_side_by_side(img_arrays):
  flatten_list = reduce(lambda x, y:x + y, zip(*img_arrays))

  plot_img_array(np.array(flatten_list), n_col=len(img_arrays))


def plot_errors(results_dict, title):
  markers = itertools.cycle(('+', 'x', 'o'))

  plt.title('{}'.format(title))

  for label, result in sorted(results_dict.items()):
    plt.plot(result, marker=next(markers), label=label)
    plt.ylabel('dice_coef')
    plt.xlabel('epoch')
    plt.legend(loc=3, bbox_to_anchor=(1, 0))

  plt.show()


def masks_to_color_img(masks):
  colors = np.asarray(
      [(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228), (56, 34, 132), (160, 194, 56)])

  colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
  channels, height, width = masks.shape

  for y in range(height):
    for x in range(width):
      selected_colors = colors[masks[:, y, x] > 0.5]

      if len(selected_colors) > 0:
        colorimg[y, x, :] = np.mean(selected_colors, axis=0)

  return colorimg.astype(np.uint8)
