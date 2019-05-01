#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc


def main():
  # lena = scipy.misc.ascent()

  # downscaling has a 'smoothing' effect
  # lena = scipy.misc.imresize(lena, 0.15, interp='cubic')

  data_set_directory = '/home/heider/Datasets/neodroid/depth/'
  file_name = '80.png'

  img = mpimg.imread(data_set_directory + file_name)
  img = img[:, :, 0]

  # def ivas(x, cam_ang):
  #  return x * math.cos(math.radians(90 - cam_ang))

  # image_length = img.shape[0]
  # camera_angle = 45.
  # ys = np.array([ivas(x, camera_angle) / 255 for x in range(0, image_length)])
  # ys = np.repeat(ys, img.shape[0], axis=0).reshape((img.shape[0], img.shape[0]))
  # img = img - ys

  img = scipy.misc.imresize(img, 0.2, interp='cubic')

  # img = img/255.

  # create the x and y coordinate arrays (here we just use pixel indices)
  d0 = img.shape[0]
  d1 = img.shape[1] / 2
  # xx, yy = np.mgrid[-d0:d0, -d1:d1]
  xx, yy = np.mgrid[0:d0, -d1:d1]

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot_surface(xx, yy, img, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)

  plt.show()


if __name__ is '__main__':
  main()
