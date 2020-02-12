#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Iterable, Sequence, List

import torch
from torch.distributions import Normal

from draugr import to_tensor
from neodroidagent.common.architectures.mlp import MLP

__author__ = "Christian Heider Nielsen"
__doc__ = ""

__all__ = ["ConcatInputMLP", "SingleHeadConcatInputMLP"]


class ConcatInputMLP(MLP):
    def __init__(self, input_shape: Sequence = (2,), **kwargs):
        if isinstance(input_shape, Iterable):
            input_shape = sum(input_shape)

        super().__init__(input_shape=input_shape, **kwargs)

    def forward(self, *x, **kwargs) -> List:
        return super().forward(torch.cat(x, dim=-1), **kwargs)


class SingleHeadConcatInputMLP(ConcatInputMLP):
    def forward(self, *x, **kwargs) -> Normal:
        return super().forward(*x, **kwargs)[0]


if __name__ == "__main__":

    def stest_normal():
        s = (10,)
        a = (10,)
        model = ConcatInputMLP(input_shape=s, output_shape=a)

        inp = to_tensor(range(s[0]), device="cpu")
        print(model.forward(inp))

    def stest_multi_dim_normal():
        s = (10, 2, 3)
        a = (2, 10)
        model = ConcatInputMLP(input_shape=s, output_shape=a)

        inp = [to_tensor(range(s_), device="cpu") for s_ in s]
        print(model.forward(*inp))

    stest_normal()
    stest_multi_dim_normal()
