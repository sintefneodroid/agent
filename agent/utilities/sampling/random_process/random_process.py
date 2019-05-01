#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC

__author__ = 'cnheider'


class RandomProcess(ABC):

  def reset(self):
    raise NotImplementedError

  def sample(self):
    raise NotImplementedError
