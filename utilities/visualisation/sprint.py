#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'

import numpy
import six

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
    )


def sprint(obj, color='white', bold=False, highlight=False):
  '''
Stylised print.
Valid colors: gray, red, green, yellow, blue, magenta, cyan, white, crimson
'''

  attr = []
  if color in color2num:
    num = color2num[color]
  else:
    num = color2num['white']
  if highlight:
    num += 10
  attr.append(six.u(str(num)))
  if bold:
    attr.append(six.u('1'))
  attrs = six.u(';').join(attr)
  print(six.u(f'\x1b[{attrs}m{obj}\x1b[0m'))


if __name__ == '__main__':
  sprint(f'{numpy.zeros(4)}, it works!', 'green')
