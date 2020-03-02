#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 24/02/2020
           """

from typing import Sequence, Tuple

import torch
from torch import nn

from neodroidagent.common.architectures.mlp import MLP

__all__ = ["DisjunctionMLP", "DuelingQMLP"]


class DisjunctionMLP(MLP):
    def __init__(
        self,
        output_shape: Sequence = (2,),
        disjunction_size=256,
        subnet_size=128,
        hidden_layer_activation=nn.ReLU(),
        **kwargs
    ):
        super().__init__(
            output_shape=(disjunction_size,),
            hidden_layer_activation=hidden_layer_activation,
            output_activation=nn.Identity(),
            **kwargs
        )

        self.subnet_1 = torch.nn.Sequential(
            torch.nn.Linear(disjunction_size, subnet_size),
            hidden_layer_activation,
            torch.nn.Linear(subnet_size, output_shape[-1]),
        )

        self.subnet_2 = torch.nn.Sequential(
            torch.nn.Linear(disjunction_size, subnet_size),
            hidden_layer_activation,
            torch.nn.Linear(subnet_size, 1),
        )

    def forward(self, *act, **kwargs) -> Tuple[torch.tensor, torch.tensor]:
        x = super().forward(*act, **kwargs)
        return self.subnet_1(x), self.subnet_2(x)


class DuelingQMLP(DisjunctionMLP):
    def forward(self, *act, **kwargs) -> torch.tensor:
        advantages, value = super().forward(*act, **kwargs)
        return value + (advantages - advantages.mean())
