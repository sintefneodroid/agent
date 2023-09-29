#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17/01/2020
           """

from typing import Sequence

import numpy
import torch
from draugr.torch_utilities import to_tensor, Architecture
from torch import nn
from torch.distributions import Categorical, MultivariateNormal

__all__ = ["ActorCritic", "CategoricalActorCritic"]


class ActorCritic(Architecture):
    def __init__(
        self,
        output_shape: Sequence = (2,),
        disjunction_size=64,
        subnet_size=64,
        hidden_layer_activation=nn.Tanh(),
        default_log_std: float = 0.6**2,
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

    def forward(self, *act, **kwargs):
        dis = super().forward(*act, **kwargs)
        return self.forward_actor(dis), self.value_subnet(dis)

    def forward_actor(self, dis, min_std=-20, max_std=2):
        act = self.policy_subnet(dis)

        # std = torch.clamp(self.log_std, min_std, max_std).exp()
        std = self.log_std.exp()

        # dist = Normal(act, scale=std.expand_as(act))

        # cov_mat = torch.diag(std).unsqueeze(dim=0)
        cov_mat = torch.diag_embed(std.expand_as(act) ** 2)
        dist = MultivariateNormal(act, covariance_matrix=cov_mat)

        return dist

    def sample(self, *act, **kwargs):
        return self.forward_actor(super().forward(*act, **kwargs)).sample()


class CategoricalActorCritic(ActorCritic):
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
        model = ActorCritic(input_shape=pos_size, output_shape=a_size)

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
        model = CategoricalActorCritic(input_shape=pos_size, output_shape=a_size)

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
