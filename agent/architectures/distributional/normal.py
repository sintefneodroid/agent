#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch.distributions import Normal
from typing import Iterable

from agent.architectures import MLP, Sequence
from agent.utilities import to_tensor
from torch.nn import functional as F

__author__ = 'cnheider'
__doc__ = ''


class MultiDimensionalNormalMLP(MLP):

  def forward(self, *x, **kwargs) -> Normal:
    out = super().forward(*x, **kwargs)

    return Normal(*out)

class NormalMLP(MLP):

  def __init__(self, output_shape: Sequence = (2,), **kwargs):
    if isinstance(output_shape, Iterable):
      if len(output_shape) != 2:
        output_shape=(output_shape[0],2)
    else:
      output_shape=(output_shape,2)
    super().__init__(output_shape=output_shape,**kwargs)


  def forward(self, *x, **kwargs) -> Normal:
    out = super().forward(*x, **kwargs)
    outs=[]
    for o in out:
      outs.append(Normal(*o))

    return outs


if __name__ == '__main__':
  def test_normal():
    s = (10,)
    a = (10,)
    model = NormalMLP(input_shape=s, output_shape=a)

    inp = to_tensor(range(s[0]))
    print(model.forward(inp)[0].sample())

  def test_multi_dim_normal():
    s = (10,)
    a = (2,10)
    model = MultiDimensionalNormalMLP(input_shape=s, output_shape=a)

    inp = to_tensor(range(s[0]))
    print(model.forward(inp).sample())

  test_normal()
  test_multi_dim_normal()