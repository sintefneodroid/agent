#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Iterable, List, Sequence

import numpy
import torch

from draugr import to_tensor
from neodroidagent.common.architectures.mlp import MLP

__author__ = "Christian Heider Nielsen"
__doc__ = ""

__all__ = ["ConcatInputMLP", "PostConcatInputMLP"]


class ConcatInputMLP(MLP):
    def __init__(self, input_shape: Sequence = (2,), **kwargs):
        if isinstance(input_shape, Iterable):
            input_shape = sum(input_shape)

        super().__init__(input_shape=input_shape, **kwargs)

    def forward(self, *x, **kwargs) -> List:
        return super().forward(torch.cat(x, dim=-1), **kwargs)


class PostConcatInputMLP(MLP):
    def __init__(
        self, input_shape: Sequence = (2, 2), output_shape: Sequence = (2,), **kwargs
    ):

        forward_shape, *res = input_shape
        self._residual_shape = res

        if not isinstance(self._residual_shape, Iterable):
            self._residual_shape = (self._residual_shape,)

        if not isinstance(output_shape, Iterable):
            output_shape = (output_shape,)

        super().__init__(
            input_shape=(forward_shape,), output_shape=output_shape, **kwargs
        )

        s = sum((*output_shape, *self._residual_shape))
        t = s * 10
        self.post_concat_layer = torch.nn.Sequential(
            torch.nn.Linear(s, t), torch.nn.ReLU(), torch.nn.Linear(t, output_shape[-1])
        )

    def forward(self, *x, **kwargs) -> torch.tensor:
        forward_x, *residual_x = x
        ax = super().forward(forward_x, **kwargs)

        return self.post_concat_layer(torch.cat((ax, *residual_x), dim=-1))


if __name__ == "__main__":

    def stest_normal():
        s = (10,)
        a = (10,)
        model = ConcatInputMLP(input_shape=s, output_shape=a)

        inp = to_tensor(range(s[0]), device="cpu")
        print(model.forward(inp))

    def stest_multi_dim_normal():
        s = (19,)
        s1 = (4,)
        batch_size = (100,)
        a = (2, 10)
        model = ConcatInputMLP(input_shape=s + s1, output_shape=a)

        inp = to_tensor(numpy.random.random(batch_size + s), device="cpu")
        inp1 = to_tensor(numpy.random.random(batch_size + s1), device="cpu")
        print(model.forward(inp, inp1))

    def stest_multi_dim_normal21():
        s = (19,)
        s1 = (4,)
        batch_size = (100,)
        a = (2, 10)
        model = PostConcatInputMLP(input_shape=s + s1, output_shape=a)

        inp = to_tensor(numpy.random.random(batch_size + s), device="cpu")
        inp1 = to_tensor(numpy.random.random(batch_size + s1), device="cpu")
        print(model.forward(inp, inp1))

    stest_normal()
    stest_multi_dim_normal()
    stest_multi_dim_normal21()
