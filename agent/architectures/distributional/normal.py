#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from typing import Iterable, List

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.distributions import Normal, MultivariateNormal
from torch.nn.functional import softplus

from agent.architectures import MLP, Sequence, xavier_init
from agent.utilities import to_tensor, fan_in_init

__author__ = 'cnheider'
__doc__ = ''


class MultiDimensionalNormalMLP(MLP):

  def __init__(self, output_shape: Sequence = (2,), **kwargs):
    if isinstance(output_shape, Iterable):
      output_shape = (2, output_shape[-1])
    else:
      output_shape = (2, output_shape)
    super().__init__(output_shape=output_shape, **kwargs)

  def forward(self, *x, **kwargs) -> Normal:
    mean, std = super().forward(*x, **kwargs)

    std = softplus(std)+1e-6
    return Normal(mean, std)

  @staticmethod
  def sample(distributions):
    with torch.no_grad():
      actions = distributions.sample()

    log_prob = distributions.log_prob(actions)

    actions = actions.to('cpu').numpy().tolist()

    return actions, log_prob

  @staticmethod
  def entropy(distributions):
    with torch.no_grad():
      return torch.mean(distributions.entropy())


class MultiVariateNormalMLP(MLP):

  def __init__(self, output_shape: Sequence = (2,), **kwargs):
    if isinstance(output_shape, Iterable):
      if len(output_shape) != 2:
        output_shape = (2, output_shape[0])
    else:
      output_shape = (2, output_shape)
    super().__init__(output_shape=output_shape, **kwargs)

  def forward(self, *x, **kwargs) -> MultivariateNormal:
    mean, std = super().forward(*x, **kwargs)

    return MultivariateNormal(mean, softplus(std)+1e-6)

  @staticmethod
  def sample(distributions):
    with torch.no_grad():
      actions = distributions.sample()

    log_prob = distributions.log_prob(actions)

    actions = actions.to('cpu').numpy().tolist()
    return actions, log_prob

  @staticmethod
  def entropy(distributions):
    with torch.no_grad():
      return torch.mean(distributions.entropy())

class MultipleNormalMLP(MLP):

  def __init__(self, output_shape: Sequence = (2,), **kwargs):
    if isinstance(output_shape, Iterable):
      if len(output_shape) != 2 or output_shape[-1] != 2:
        output_shape = (output_shape[0], 2)
    else:
      output_shape = (output_shape, 2)
    super().__init__(output_shape=output_shape, **kwargs)

    fan_in_init(self)

  @staticmethod
  def sample(distributions):
    with torch.no_grad():
      actions = [d.sample() for d in distributions]

    log_prob = [d.log_prob(action) for d, action in zip(distributions, actions)]

    actions = [a.to('cpu').numpy().tolist() for a in actions]
    return actions, log_prob

  @staticmethod
  def entropy(distributions):
    with torch.no_grad():
      return torch.mean(to_tensor([d.entropy() for d in distributions]))

  def forward(self, *x, **kwargs) -> List[Normal]:
    out = super().forward(*x, **kwargs)
    outs = []
    for a in out:
      if a.shape[0] == 2:
        mean, std = a
      else:
        mean, std = a[0]
      outs.append(Normal(mean, softplus(std)+1e-6))

    return outs


if __name__ == '__main__':
  def test_normal():
    s = (10,)
    a = (10)
    model = MultipleNormalMLP(input_shape=s, output_shape=a)

    inp = torch.rand(s)
    s_ = time.time()
    a_ =model.sample(model.forward(inp))
    print(time.time()-s_, a_)


  def test_multi_dim_normal():
    s = (4,)
    a = (10,)
    model = MultiDimensionalNormalMLP(input_shape=s, output_shape=a)

    inp = torch.rand(s)
    s_ = time.time()
    a_ =model.sample(model.forward(inp))
    print(time.time()-s_, a_)

  def test_multi_dim_normal_2():
    s = (1,4)
    a = (1,10)
    model = MultiDimensionalNormalMLP(input_shape=s, output_shape=a)

    inp = torch.rand(s)
    s_ = time.time()
    a_ =model.sample(model.forward(inp))
    print(time.time()-s_, a_)

  def test_multi_var_normal():
    s = (10,)
    a = (10)
    model = MultiVariateNormalMLP(input_shape=s, output_shape=a)

    inp = torch.rand(s)
    s_ = time.time()
    a_ =model.sample(model.forward(inp))
    print(time.time()-s_, a_)


  test_normal()
  print('\n')
  test_multi_dim_normal()
  print('\n')
  test_multi_dim_normal_2()
  #test_multi_var_normal()
