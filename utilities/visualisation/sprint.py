#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'

import numpy
import six

colors = dict(
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

decorations = dict(
    end=0,
    bold=1,
    underlined=4
    )


def sprint(obj, **kwargs):
  '''
Stylised print.
Valid colors: gray, red, green, yellow, blue, magenta, cyan, white, crimson
'''

  string = style(obj,**kwargs)

  print(string)

def style(obj=None, *,
           color='white',
           bold=False,
           highlight=False,
           underlined=False):
  attributes = []

  if color in colors:
    num = colors[color]
  else:
    num = colors['white']

  if highlight:
    num += 10

  attributes.append(six.u(str(num)))

  if bold:
    attributes.append(six.u(str(decorations['bold'])))

  if underlined:
    attributes.append(six.u(str(decorations['underlined'])))

  end = decorations["end"]

  attributes_joined = six.u(';').join(attributes)

  if obj:
    intermediate_repr = f'\x1b[{attributes_joined}m{obj}\x1b[{end}m'
    string = six.u(intermediate_repr)

    return string
  else:
    return PrintStyle(attributes_joined,end)

class PrintStyle(object):
  def __init__(self, attributes_joined, end):
    self._attributes_joined = attributes_joined
    self._end = end

  def __call__(self, obj, *args, **kwargs):
    intermediate_repr = f'\x1b[{self._attributes_joined}m{obj}\x1b[{self._end}m'
    string = six.u(intermediate_repr)



if __name__ == '__main__':
  sprint(f'{numpy.zeros(4)}, it works!', color='green')
