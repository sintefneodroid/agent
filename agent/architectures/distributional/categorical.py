#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy import prod
from torch.distributions import Categorical
from torch.nn import functional as F

from agent.architectures import MLP, numpy, torch
from agent.utilities import to_tensor

__author__ = 'cnheider'
__doc__ = ''


class MultipleCategoricalMLP(MLP):

  @staticmethod
  def sample(distributions):
    actions = [d.sample() for d in distributions][0]

    log_prob = [d.log_prob(action) for d, action in zip(distributions, actions)][0]

    actions = [a.to('cpu').numpy().tolist() for a in actions]
    return actions, log_prob

  @staticmethod
  def entropy(distributions):
    return torch.mean(to_tensor([d.entropy() for d in distributions]))

  def forward(self, *x, **kwargs):
    out = super().forward(*x, **kwargs)
    outs = []
    for o in out:
      outs.append(Categorical(F.softmax(o, dim=-1)))

    return outs


class CategoricalMLP(MLP):

  @staticmethod
  def sample(distributions):
    actions = distributions.sample()

    log_prob = distributions.log_prob(actions)

    actions = actions.to('cpu').numpy().tolist()
    return actions, log_prob

  @staticmethod
  def entropy(distributions):
    return distributions.entropy()

  def forward(self, *x, **kwargs):
    out = super().forward(*x, **kwargs)[0]
    return Categorical(F.softmax(out, dim=-1))


if __name__ == '__main__':
  def multi_cat():
    s = (2, 2)
    a = (2, 2)
    model = MultipleCategoricalMLP(input_shape=s, output_shape=a)

    inp = to_tensor(numpy.random.rand(64, prod(s[1:])))
    print(model.sample(model(inp, inp)))


  def single_cat():
    s = (1, 2)
    a = (1, 2)
    model = CategoricalMLP(input_shape=s, output_shape=a)

    inp = to_tensor(numpy.random.rand(64, prod(s[1:])))
    print(model.sample(model(inp)))


  def single_cat2():
    s = (4,)
    a = (2,)
    model = CategoricalMLP(input_shape=s, output_shape=a)

    inp = to_tensor(numpy.random.rand(64, prod(s)))
    print(model.sample(model(inp)))

  multi_cat()
  single_cat()
  single_cat2()