#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17/01/2020
           """

__all__ = ["ActorCriticFissionMLP", "CategoricalActorCriticFissionMLP"]

from typing import Sequence

import numpy
import torch
from draugr.torch_utilities import MLP, to_tensor, ortho_init
from torch import nn
from torch.distributions import Categorical, Normal, MultivariateNormal


# TODO:  Beta distribution then use ReLU and for Normal distribution tanh activation


class ActorCriticFissionMLP(MLP):
    def __init__(
        self,
        output_shape: Sequence = (2,),
        disjunction_size=64,
        subnet_size=64,
        hidden_layer_activation=nn.Tanh(),
        default_log_std: float = -0.5,
        default_init=ortho_init,
        **kwargs
    ):
        super().__init__(
            output_shape=(disjunction_size,),
            default_log_std=default_log_std,
            hidden_layer_activation=hidden_layer_activation,
            default_init=default_init,
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

    def forward(self, *act, **kwargs):
        x = super().forward(*act, **kwargs)
        return self.forward_actor(x), self.value_subnet(x)

    def forward_actor(
        self,
        x,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        clamp_log_std: bool = True,
    ):
        mean_ = self.policy_subnet(x)

        if clamp_log_std:
            std = torch.clamp(self.log_std, min_log_std, max_log_std).exp()
        else:
            std = self.log_std.exp()

        # sampling_distribution = Normal(act, scale=std.expand_as(act))
        sampling_distribution = MultivariateNormal(
            mean_, covariance_matrix=torch.diag_embed(std.expand_as(mean_))
        )

        return sampling_distribution

    def sample(self, *act, **kwargs):
        return self.forward_actor(super().forward(*act, **kwargs)).sample()


class CategoricalActorCriticFissionMLP(ActorCriticFissionMLP):
    def __init__(self, output_shape: Sequence = (2,), **kwargs):
        aug_output_shape = (*output_shape, 1)
        super().__init__(output_shape=aug_output_shape, **kwargs)

    def forward_actor(self, dis, **kwargs):
        return Categorical(torch.softmax(dis, dim=-1))


if __name__ == "__main__":

    def stest_single_dim():
        pos_size = (4,)
        a_size = (4,)
        batch_size = 64
        model = ActorCriticFissionMLP(input_shape=pos_size, output_shape=a_size)

        print(
            torch.mean(
                to_tensor(
                    [
                        model(
                            to_tensor(
                                numpy.random.rand(batch_size, pos_size[0]),
                                device="cpu",
                                dtype=torch.float,
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
        model = CategoricalActorCriticFissionMLP(
            input_shape=pos_size, output_shape=a_size
        )

        print(
            torch.mean(
                to_tensor(
                    [
                        model(
                            to_tensor(
                                numpy.random.rand(batch_size, pos_size[0]),
                                device="cpu",
                                dtype=torch.float,
                            )
                        )[0].sample()
                        for _ in range(1000)
                    ],
                )
            )
        )

    stest_single_dim_cat()
    stest_single_dim()
