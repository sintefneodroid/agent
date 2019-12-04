#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC

from torch import nn

__author__ = "Christian Heider Nielsen"


class Architecture(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def __repr__(self):
        num_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        num_params = sum(param.numel() for param in self.parameters())

        return f"{super().__repr__()}\ntrainable/num_params: {num_trainable_params}/{num_params}\n"
