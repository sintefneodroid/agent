#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Iterable, Sequence

import torch
from torch.distributions import Normal

from neodroidagent.architectures import MLP, to_tensor

__author__ = 'cnheider'
__doc__ = ''


class MergedInputMLP(MLP):

  def __init__(self, input_shape: Sequence = (2,), **kwargs):
    if isinstance(input_shape, Iterable):
      input_shape = sum(input_shape)

    super().__init__(input_shape=input_shape, **kwargs)

  def forward(self, *x, **kwargs) -> Normal:
    out = super().forward(torch.cat(x, dim=-1), **kwargs)
    return out


class SingleHeadMergedInputMLP(MergedInputMLP):
  def forward(self, *x, **kwargs) -> Normal:
    out = super().forward(*x, **kwargs)
    return out[0]


if __name__ == '__main__':
  def test_normal():
    s = (10,)
    a = (10,)
    model = MergedInputMLP(input_shape=s, output_shape=a)

    inp = to_tensor(range(s[0]))
    print(model.forward(inp))


  def test_multi_dim_normal():
    s = (10, 2, 3)
    a = (2, 10)
    model = MergedInputMLP(input_shape=s, output_shape=a)

    inp = [to_tensor(range(s_)) for s_ in s]
    print(model.forward(*inp))


  test_normal()
  test_multi_dim_normal()
