#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy import prod
from torch.distributions import Categorical
from torch.nn import functional as F

from draugr import to_tensor
from neodroidagent.common.architectures import MLP
import numpy
import torch

__author__ = "Christian Heider Nielsen"
__doc__ = ""

__all__ = ["MultipleCategoricalMLP", "CategoricalMLP"]


class MultipleCategoricalMLP(MLP):
    @staticmethod
    def sample(distributions):
        actions = [d.sample() for d in distributions][0]

        log_prob = [d.log_prob(action) for d, action in zip(distributions, actions)][0]

        actions = [a.to("cpu").numpy().tolist() for a in actions]
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
    def forward(self, *x, **kwargs):
        out = super().forward(*x, **kwargs)[0]
        return Categorical(F.softmax(out, dim=-1))


if __name__ == "__main__":

    def multi_cat():
        s = (2, 2)
        a = (2, 2)
        model = MultipleCategoricalMLP(input_shape=s, output_shape=a)

        inp = to_tensor(numpy.random.rand(64, prod(s[1:])), device="cpu")
        print(model.sample(model(inp, inp)))

    def single_cat():
        s = (1, 2)
        a = (1, 2)
        model = CategoricalMLP(input_shape=s, output_shape=a)

        inp = to_tensor(numpy.random.rand(64, prod(s[1:])), device="cpu")
        print(model.sample(model(inp)))

    def single_cat2():
        s = (4,)
        a = (2,)
        model = CategoricalMLP(input_shape=s, output_shape=a)

        inp = to_tensor(numpy.random.rand(64, prod(s)), device="cpu")
        print(model.sample(model(inp)))

    multi_cat()
    single_cat()
    single_cat2()
