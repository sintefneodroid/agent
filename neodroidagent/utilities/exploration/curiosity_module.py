#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABCMeta

from torch import nn

__author__ = 'cnheider'
__doc__ = r'''
'''


class CuriosityMeta(metaclass=ABCMeta):
  pass

class CuriosityModule(CuriosityMeta, nn.Module):
  pass

