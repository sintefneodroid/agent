#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'


from .consise_arch_spec import *
from .optimiser_spec import *

class HyperParameters:
  """
  This class is friendly wrapper over Python Dictionary
  to represent the named hyperparameters.

  Example:

      One can manually set arbitrary strings as hyperparameters as

      .. code-block:: python

            hparams = HyperParameters()
            hparams.paramA = 'myparam'
            hparams.paramB = 10

      or just send in a dictionary object containing all the relevant key/value
      pairs.

      .. code-block:: python

            hparams = HyperParameters({'paramA': 'myparam', 'paramB': 10})
            assert hparams.paramA == 'myparam'
            assert hparams.paramB == 10

      Both form equivalent hyperparameter objects.

      To update/override the hyperparamers, use the `update()` method.

      .. code-block:: python

          hparams.update({'paramA': 20, 'paramB': 'otherparam', 'paramC': 5.0})
          assert hparams.paramA == 20
          assert hparams.paramB == 'otherparam'

  Args:
    kwargs (dict): Python dictionary representing named hyperparameters and
    values.

  """
  def __init__(self, kwargs=None):
    self.update(kwargs or {})

  def __getattr__(self, item):
    return self.__dict__[item]

  def __setattr__(self, key, value):
    self.__dict__[key] = value

  def __iter__(self):
    for key, value in self.__dict__.items():
      yield key, value

  def __repr__(self):
    print_str = ''
    for key, value in self:
      print_str += '{}: {}\n'.format(key, value)
    return print_str

  def update(self, items: dict):
    """
    Merge two Hyperparameter objects, overriding any repeated keys from
    the `items` parameter.

    Args:
        items (dict): Python dictionary containing updated values.
    """
    self.__dict__.update(items)