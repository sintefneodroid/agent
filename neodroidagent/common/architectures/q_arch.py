#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """
__all__ = ["DuelingQMLP"]

import torch
from draugr.torch_utilities import DisjunctMLP, ortho_init

from torch import nn


class DuelingQMLP(DisjunctMLP):
    def __init__(
        self, hidden_layer_activation=nn.Tanh(), default_init=ortho_init, **kwargs
    ):
        super().__init__(
            hidden_layer_activation=hidden_layer_activation,
            default_init=default_init,
            **kwargs
        )

    def forward(self, *act, **kwargs) -> torch.tensor:
        """

        :param act:
        :type act:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        advantages, value = super().forward(*act, **kwargs)
        return value + (advantages - advantages.mean())
