#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17/01/2020
           """

from neodroidagent.common.architectures.mlp import MLP

__all__ = ["SingleHeadMLP"]


class SingleHeadMLP(MLP):
    def forward(self, *x, **kwargs):
        outs = super().forward(*x, **kwargs)
        return outs[0]
