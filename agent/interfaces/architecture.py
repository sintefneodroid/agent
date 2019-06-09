#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC

from torch import nn

__author__ = 'cnheider'


class Architecture(nn.Module, ABC):

  def __init__(self, **kwargs):
    super().__init__()

  @property
  def input_shape(self):
    return self._input_shape

  @property
  def output_shape(self):
    return self._output_shape
