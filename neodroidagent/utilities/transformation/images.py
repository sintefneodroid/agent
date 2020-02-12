#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy import prod

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """

import numpy

__all__ = ["tile_images"]


def tile_images(img_nhwc):
    """
Tile N images into one big PxQ image
(P,Q) are chosen to be as close as possible, and if N
is square, then P=Q.

:param img_nhwc: (list) list or array of images, ndim=4 once turned into array. img nhwc
  n = batch index, h = height, w = width, c = channel
:return: (numpy float) img_HWc, ndim=3
"""

    img_nhwc = numpy.asarray(img_nhwc)
    n_images, height, width, n_channels = img_nhwc.shape
    # new_height was named H before
    new_height = int(numpy.ceil(numpy.sqrt(n_images)))
    # new_width was named W before
    new_width = int(numpy.ceil(float(n_images) / new_height))
    img_nhwc = numpy.array(
        list(img_nhwc)
        + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)]
    )
    # img_HWhwc
    out_image = img_nhwc.reshape(new_height, new_width, height, width, n_channels)
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape(new_height * height, new_width * width, n_channels)
    return out_image


if __name__ == "__main__":
    import cv2

    def sigmoid(x: numpy.array, derivative: bool = False) -> numpy.array:
        sigm = 1.0 / (1.0 + numpy.exp(-x))
        if derivative:
            return sigm * (1.0 - sigm)
        return sigm

    s = (4, 8, 8, 3)
    a = numpy.array(range(int(prod(s))))
    a = a * 0.003
    a = sigmoid(a)
    a = a.reshape(s)
    r = tile_images(a)
    print(r)
    cv2.imshow("a", r[:, :, ::-1])
    cv2.waitKey()
