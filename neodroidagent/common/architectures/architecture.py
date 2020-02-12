#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC

from torch import nn

__author__ = "Christian Heider Nielsen"

__all__ = ["Architecture"]

from warg import drop_unused_kws
from draugr import add_indent


class Architecture(nn.Module, ABC):
    @drop_unused_kws
    def __init__(self):
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

        di = add_indent(f"{self.__dict__}")

        trainable_params_str = add_indent(
            f"trainable/num_params: {num_trainable_params}/{num_params}\n"
        )

        return f"{super().__repr__()}\n{di}\n{trainable_params_str}"


if __name__ == "__main__":
    a = Architecture()

    print(a)
