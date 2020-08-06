#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04/12/2019
           """

from draugr.torch_utilities import to_tensor
from neodroidagent.common.architectures.mlp_variants import PreConcatInputMLP


def test_normal():
    s = (10,)
    a = (10,)
    model = PreConcatInputMLP(input_shape=s, output_shape=a)

    inp = to_tensor(range(s[0]), device="cpu")
    print(model.forward(inp))


def test_multi_dim_normal():
    s = (10, 2, 3)
    a = (2, 10)
    model = PreConcatInputMLP(input_shape=s, output_shape=a)

    inp = [to_tensor(range(s_), device="cpu") for s_ in s]
    print(model.forward(*inp))


if __name__ == "__main__":
    test_normal()
    test_multi_dim_normal()
