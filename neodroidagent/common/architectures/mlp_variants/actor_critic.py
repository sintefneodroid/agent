#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17/01/2020
           """

from typing import Sequence

import numpy
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from draugr import to_tensor
from neodroidagent.common.architectures.mlp import MLP

__all__ = ["ActorCriticMLP", "CategoricalActorCriticMLP"]


class ActorCriticMLP(MLP):
    def __init__(
        self,
        output_shape: Sequence = (2,),
        disjunction_size=256,
        subnet_size=128,
        hidden_layer_activation=nn.ReLU(),
        default_log_std: float = 0,
        **kwargs
    ):
        super().__init__(
            output_shape=(disjunction_size,),
            default_log_std=default_log_std,
            hidden_layer_activation=hidden_layer_activation,
            output_activation=nn.Identity(),
            **kwargs
        )

        self.policy_subnet = torch.nn.Sequential(
            torch.nn.Linear(disjunction_size, subnet_size),
            hidden_layer_activation,
            torch.nn.Linear(subnet_size, output_shape[-1]),
        )

        self.value_subnet = torch.nn.Sequential(
            torch.nn.Linear(disjunction_size, subnet_size),
            hidden_layer_activation,
            torch.nn.Linear(subnet_size, 1),
        )

        self.log_std = nn.Parameter(
            torch.ones(output_shape[-1]) * default_log_std, requires_grad=True
        )

    def forward(self, *act, min_std=-20, max_std=2, **kwargs):
        dis = super().forward(*act, min_std=min_std, **kwargs)

        act = self.policy_subnet(dis)
        val = self.value_subnet(dis)
        std = torch.clamp(self.log_std, min_std, max_std).exp().expand_as(act)
        return Normal(torch.tanh(act), std), val


class CategoricalActorCriticMLP(MLP):
    def __init__(self, output_shape: Sequence = (2,), **kwargs):
        aug_output_shape = (*output_shape, 1)
        super().__init__(output_shape=aug_output_shape, **kwargs)

    def forward(self, *act, **kwargs):
        act, val = super().forward(*act, **kwargs)
        return Categorical(torch.softmax(act, dim=-1)), val


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
                        for _ in range(1000)
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
    stest_single_dim()
