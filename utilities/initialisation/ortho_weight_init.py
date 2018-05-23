#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

import numpy as np
import torch


def ortho_weights(shape, scale=1.):
  ''' PyTorch port of ortho_init from baselines.a2c.utils '''
  shape = tuple(shape)

  if len(shape) == 2:
    flat_shape = shape[1], shape[0]
  elif len(shape) == 4:
    flat_shape = (np.prod(shape[1:]), shape[0])
  else:
    raise NotImplementedError

  a = np.random.normal(0., 1., flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  q = q.transpose().copy().reshape(shape)

  if len(shape) == 2:
    return torch.from_numpy((scale * q).astype(np.float32))
  if len(shape) == 4:
    return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))
