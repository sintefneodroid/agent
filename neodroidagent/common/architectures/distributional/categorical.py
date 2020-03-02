#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Tuple

import numpy
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from draugr import to_tensor
from neodroidagent.common.architectures.mlp import MLP

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""
__all__ = ["MultipleCategoricalMLP", "CategoricalMLP"]


class MultipleCategoricalMLP(MLP):
    @staticmethod
    def sample(distributions) -> Tuple:
        actions = [d.sample() for d in distributions][0]

        log_prob = [d.log_prob(action) for d, action in zip(distributions, actions)][0]

        actions = [a.to("cpu").numpy().tolist() for a in actions]
        return actions, log_prob

    @staticmethod
    def entropy(distributions) -> torch.tensor:
        return torch.mean(to_tensor([d.entropy() for d in distributions]))

    def forward(self, *x, **kwargs) -> List:
        out = super().forward(*x, **kwargs)
        outs = []
        for o in out:
            outs.append(Categorical(F.softmax(o, dim=-1)))

        return outs


class CategoricalMLP(MLP):
    def forward(self, *x, **kwargs) -> Categorical:
        return Categorical(F.softmax(super().forward(*x, **kwargs), dim=-1))


if __name__ == "__main__":

    def multi_cat():
        s = (2, 2)
        a = (2, 2)
        model = MultipleCategoricalMLP(input_shape=s, output_shape=a)

        inp = to_tensor(numpy.random.rand(64, s[0]), device="cpu")
        print(model.sample(model(inp, inp)))

    def single_cat():
        s = (1, 2)
        a = (2,)
        model = CategoricalMLP(input_shape=s, output_shape=a)

        inp = to_tensor(numpy.random.rand(64, s[0]), device="cpu")
        inp2 = to_tensor(numpy.random.rand(64, s[1]), device="cpu")
        print(model(inp, inp2).sample())

    def single_cat2():
        s = (4,)
        a = (2,)
        model = CategoricalMLP(input_shape=s, output_shape=a)

        inp = to_tensor(numpy.random.rand(64, s[0]), device="cpu")
        print(model(inp).sample())

    multi_cat()
    single_cat()
    single_cat2()
