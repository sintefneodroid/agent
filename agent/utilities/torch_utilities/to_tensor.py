#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sequence, Iterable, Union

import numpy
import torch

__author__ = 'cnheider'
__doc__ = ''

def to_tensor(obj: Union[torch.Tensor, numpy.ndarray, Iterable, int, float], dtype=torch.float, device='cpu',
              non_blocking=True):
  if torch.is_tensor(obj):
    return obj.to(device, dtype=dtype, non_blocking=non_blocking)

  if isinstance(obj, numpy.ndarray):
    if torch.is_tensor(obj[0]) and len(obj[0].size()) > 0:
      return torch.stack(obj.tolist())
    return torch.from_numpy(obj).to(device=device,
                                    dtype=dtype,
                                    non_blocking=non_blocking)

  if not isinstance(obj, Sequence):
    obj = [obj]
  elif not isinstance(obj, list) and isinstance(obj, Iterable):
    obj = [*obj]


  if isinstance(obj, list):
    if torch.is_tensor(obj[0]) and len(obj[0].size()) > 0:
      return torch.stack(obj)


  return torch.tensor(obj, device=device, dtype=dtype)


if __name__ == '__main__':
  print(to_tensor(1))
  print(to_tensor(2.0))
  print(to_tensor([0.5, 0.5]))
  print(to_tensor([[0.5, 0.5]]))
  print(to_tensor((0.5, 0.5)))
  print(to_tensor(range(10)))
  print(to_tensor(torch.from_numpy(numpy.array([0.5, 0.5]))))
  a = torch.arange(0,10)
  print(to_tensor(a))
  print(to_tensor([a,a]))
  print(to_tensor((a, a)))
