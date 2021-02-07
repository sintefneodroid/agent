#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy

from draugr.torch_utilities import to_tensor
from neodroidagent.common.architectures import MLP
import torch
import pytest

__author__ = "Christian Heider Nielsen"
__doc__ = ""


def test_single_dim():
    pos_size = (4,)
    a_size = (1,)
    model = MLP(input_shape=pos_size, output_shape=a_size)

    pos_1 = to_tensor(
        numpy.random.rand(64, pos_size[0]), device="cpu", dtype=torch.float
    )
    print(model(pos_1))


def test_hidden_dim():
    pos_size = (4,)
    hidden_size = (2, 3)
    a_size = (2,)
    model = MLP(input_shape=pos_size, hidden_layers=hidden_size, output_shape=a_size)

    pos_1 = to_tensor(
        numpy.random.rand(64, pos_size[0]), device="cpu", dtype=torch.float
    )
    print(model(pos_1))


@pytest.mark.skip
def test_multi_dim():
    """
    TODO: BROKEN!
    """

    pos_size = (2, 3, 2)  # two, 2d tensors
    a_size = (2, 4, 5)
    model = MLP(input_shape=pos_size, output_shape=a_size)

    pos_1 = to_tensor(
        numpy.random.rand(64, numpy.prod(pos_size[1:])), device="cpu", dtype=torch.float
    )
    pos_2 = to_tensor(
        numpy.random.rand(64, numpy.prod(pos_size[1:])), device="cpu", dtype=torch.float
    )
    print(model(pos_1, pos_2))


if __name__ == "__main__":
    test_single_dim()
    test_hidden_dim()
    # test_multi_dim()
