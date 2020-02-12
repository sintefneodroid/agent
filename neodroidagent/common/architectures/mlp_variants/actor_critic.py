#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17/01/2020
           """

from typing import Iterable, Sequence

import numpy
import torch
from torch import nn
from torch.distributions import Normal, Categorical
from torch.nn.functional import softplus

from draugr import to_tensor
from neodroidagent.common.architectures.mlp import MLP

__all__ = ["ActorCriticMLP", "CategoricalActorCriticMLP"]


class ActorCriticMLP(MLP):
    def __init__(self, output_shape: Sequence = (2,), default_std=1e-3, **kwargs):
        if not isinstance(output_shape, Iterable):
            output_shape = (1, output_shape)
        super().__init__(output_shape=output_shape, default_std=default_std, **kwargs)

        self.log_std = nn.Parameter(
            torch.ones(output_shape[-1]) * default_std, requires_grad=True
        )
        self.value_out = nn.Linear(output_shape[-1], 1)

    def forward(self, *x, min_std=-20, max_std=2, **kwargs):
        x = super().forward(*x, min_std=min_std, **kwargs)[0]
        std = torch.clamp(self.log_std, min_std, max_std).exp().expand_as(x)
        return Normal(x, std), self.value_out(x)


class CategoricalActorCriticMLP(MLP):
    def __init__(self, output_shape: Sequence = (2,), **kwargs):
        if not isinstance(output_shape, Iterable):
            output_shape = (1, output_shape)

        super().__init__(output_shape=output_shape, **kwargs)

        self.value_out = nn.Linear(output_shape[-1], 1)

    def forward(self, *x, **kwargs):
        x = super().forward(*x, **kwargs)[0]
        return Categorical(torch.softmax(x, dim=-1)), self.value_out(x)


if __name__ == "__main__":

    def stest_single_dim():
        pos_size = (4,)
        a_size = (4,)
        batch_size = 64
        model = ActorCriticMLP(input_shape=pos_size, output_shape=a_size)

        print(
            torch.mean(
                to_tensor(
                    [
                        model(
                            to_tensor(
                                numpy.random.rand(batch_size, pos_size[0]), device="cpu"
                            )
                        )[0].sample()
                        for _ in range(10000)
                    ]
                )
            )
        )

    def stest_single_dim_cat():
        pos_size = (4,)
        a_size = (2,)
        batch_size = 64
        model = CategoricalActorCriticMLP(input_shape=pos_size, output_shape=a_size)

        print(
            torch.mean(
                to_tensor(
                    [
                        model(
                            to_tensor(
                                numpy.random.rand(batch_size, pos_size[0]), device="cpu"
                            )
                        )[0].sample()
                        for _ in range(1000)
                    ]
                )
            )
        )

    stest_single_dim_cat()
    # stest_single_dim()
