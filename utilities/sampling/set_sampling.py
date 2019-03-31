#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import typing
from typing import Any

__author__ = 'cnheider'

import numpy as np


def sample(set: iter) -> Any:
  a = list(set)
  if len(a):
    idx = np.random.randint(0, len(a))
    return a[idx]
