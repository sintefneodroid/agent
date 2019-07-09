#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy import prod
from torch.distributions import Categorical

from agent.architectures import MLP, numpy, Sequence
from agent.utilities import to_tensor, Iterable
from torch.nn import functional as F

__author__ = 'cnheider'
__doc__ = ''

class CategoricalMLP(MLP):

  def forward(self, *x, **kwargs):
    out = super().forward(*x, **kwargs)
    outs=[]
    for o in out:
      outs.append(Categorical(F.softmax(o, dim=-1)))

    return outs


if __name__ == '__main__':
    s = (2,2)
    a = (2,2)
    model = CategoricalMLP(input_shape=s, output_shape=a)

    inp = to_tensor(numpy.random.rand(64,prod(s[1:])))
    print(model(inp,inp))