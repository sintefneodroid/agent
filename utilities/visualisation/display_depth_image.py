#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main():
  data_set_directory = '/home/heider/Datasets/neodroid/'
  file_name = '10.png'

  # img = Image.open(data_set_directory + file_name).convert('LA')
  # img_array = np.asarray(img)
  # print(img_array.shape)

  img = mpimg.imread(data_set_directory + file_name)
  img = img[:, :, 0]
  img = img / 255

  plt.imshow(img, cmap=plt.get_cmap('gray'))
  plt.show()


if __name__ is '__main__':
  main()
