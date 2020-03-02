#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from typing import Sequence, Sized

import numpy
import torch
from numpy import prod
from torch import nn
from torch.nn import Module

from draugr import constant_init, fan_in_init, to_tensor, torch_seed
from neodroidagent.common.architectures.architecture import Architecture
from warg.named_ordered_dictionary import NOD

__author__ = "Christian Heider Nielsen"

__doc__ = r"""
Description: Multi Layer Perceptron
Author: Christian Heider Nielsen
"""

__all__ = ["MLP"]


class MLP(Architecture):
    """
OOOO input_shape
|XX|                                        fc1
OOOO hidden_layer_size * (Weights,Biases)
|XX|                                        fc2
OOOO hidden_layer_size * (Weights,Biases)
|XX|                                        fc3
0000 output_shape * (Weights,Biases)
"""

    def __init__(
        self,
        *,
        input_shape: Sized = None,
        hidden_layers: Sized = None,
        hidden_layer_activation: Module = torch.nn.ReLU(),
        output_shape: Sized = None,
        output_activation: Module = torch.nn.Identity(),
        use_bias: bool = True,
        use_dropout: bool = False,
        dropout_prob: float = 0.2,
        auto_build_hidden_layers_if_none=True,
        input_multiplier: int = 32,
        max_layer_width=1000,
        output_multiplier: int = 16,
        default_init: callable = fan_in_init,
        # prefix:str=None, #TODO name sub networks
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert input_shape is not None
        assert output_shape is not None

        self._input_shape = None
        self._output_shape = None

        self.infer_input_shape(input_shape)
        self.infer_output_shape(output_shape)

        self._use_bias = use_bias

        if not hidden_layers and auto_build_hidden_layers_if_none:
            hidden_layers = self.construct_progressive_hidden_layers(
                self._input_shape,
                self._output_shape,
                input_multiplier,
                output_multiplier,
                max_layer_width=max_layer_width,
            )
        elif isinstance(hidden_layers, int):
            hidden_layers = (hidden_layers,)

        if use_dropout:
            HiddenLinear = lambda *a, **b: nn.Sequential(
                nn.Linear(*a, **b), nn.Dropout(p=dropout_prob), hidden_layer_activation
            )
        else:
            HiddenLinear = lambda *a, **b: nn.Sequential(
                nn.Linear(*a, **b), hidden_layer_activation
            )

        self._hidden_layers = hidden_layers
        self.num_of_layer = len(self._hidden_layers) - 1
        assert self.num_of_layer >= 0

        for i, siz in enumerate(self._input_shape):
            setattr(
                self,
                f"_in{i}",
                HiddenLinear(siz, self._hidden_layers[0], bias=self._use_bias),
            )

        previous_layer_size = self._hidden_layers[0] * len(self._input_shape)

        for i in range(self.num_of_layer):
            setattr(
                self,
                f"_hidden{i}",
                HiddenLinear(
                    previous_layer_size, self._hidden_layers[i], bias=self._use_bias
                ),
            )
            previous_layer_size = self._hidden_layers[i]

        for i, siz in enumerate(self._output_shape):
            setattr(
                self,
                f"_out{i}",
                nn.Sequential(
                    nn.Linear(previous_layer_size, siz, bias=self._use_bias),
                    output_activation,
                ),
            )

        if default_init:
            default_init(self)

    @staticmethod
    def construct_progressive_hidden_layers(
        _input_shape,
        _output_shape,
        input_multiplier: float = 32,
        output_multiplier: float = 16,
        max_layer_width: int = 1000,
    ):
        h_first_size = min(int(sum(_input_shape) * input_multiplier), max_layer_width)
        h_last_size = min(int(sum(_output_shape) * output_multiplier), max_layer_width)

        h_middle_size = int(numpy.sqrt(h_first_size * h_last_size))

        hidden_layers = NOD(h_first_size, h_middle_size, h_last_size).as_list()

        return hidden_layers

    def infer_input_shape(self, input_shape):
        """

@param input_shape:
@return:
"""
        if isinstance(input_shape, Sequence):
            assert len(input_shape) > 0, f"Got length {len(input_shape)}"
            self._input_shape = input_shape
        elif isinstance(input_shape, int):
            self._input_shape = (input_shape,)
            logging.info(
                f"Inflating input shape {input_shape} to vectorised input shape {self._input_shape}"
            )
        else:
            raise ValueError(f"Can not use {input_shape} as input shape")

    def infer_output_shape(self, output_shape):
        """

@param output_shape:
@return:
"""
        if isinstance(output_shape, Sequence):
            assert len(output_shape) > 0, f"Got length {len(output_shape)}"
            self._output_shape = output_shape
        elif isinstance(output_shape, int):
            self._output_shape = (output_shape,)
            logging.info(
                f"Inflating output shape {output_shape} to vectorised output shape {self._output_shape}"
            )
        else:
            raise ValueError(f"Can not use {output_shape} as output shape")

    def forward(self, *x, **kwargs):
        """

:param x:
:return output:
"""
        if len(x) != len(self.input_shape):
            raise ValueError(
                f"{len(self.input_shape)} input arguments expected, {len(x)} was supplied"
            )

        ins = []
        for i in range(len(self._input_shape)):
            ins.append(getattr(self, f"_in{i}")(x[i]))

        val = torch.cat(ins, dim=-1)
        for i in range(self.num_of_layer):
            val = getattr(self, f"_hidden{i}")(val)

        outs = []
        for i in range(len(self._output_shape)):
            outs.append(getattr(self, f"_out{i}")(val))

        if len(outs) == 1:
            return outs[0]

        return outs


if __name__ == "__main__":
    torch_seed(4)

    def stest_single_dim():
        pos_size = (4,)
        a_size = (1,)
        model = MLP(input_shape=pos_size, output_shape=a_size)

        pos_1 = to_tensor(numpy.random.rand(64, pos_size[0]), device="cpu")
        print(model(pos_1)[0].shape)

    def stest_hidden_dim():
        pos_size = (3,)
        hidden_size = list(range(6, 10))
        a_size = (4,)
        model = MLP(
            input_shape=pos_size,
            hidden_layers=hidden_size,
            output_shape=a_size,
            hidden_layer_activation=torch.tanh,
            default_init=None,
        )

        model2 = nn.Sequential(
            *[
                nn.Linear(3, 6),
                nn.Tanh(),
                nn.Linear(6, 7),
                nn.Tanh(),
                nn.Linear(7, 8),
                nn.Tanh(),
                nn.Linear(8, 9),
                nn.Tanh(),
                nn.Linear(9, 4),
            ]
        )
        model3 = nn.Sequential(
            *[
                nn.Linear(3, 6),
                nn.Tanh(),
                nn.Linear(6, 7),
                nn.Tanh(),
                nn.Linear(7, 8),
                nn.Tanh(),
                nn.Linear(8, 9),
                nn.Tanh(),
                nn.Linear(9, 4),
            ]
        )
        constant_init(model, 0.142)
        constant_init(model2, 0.142)
        constant_init(model3, 0.142)
        print(model, model2, model3)

        pos_1 = to_tensor(numpy.random.rand(64, pos_size[0]), device="cpu")
        print(model(pos_1)[0].shape)
        print(model2(pos_1).shape)
        print(model3(pos_1).shape)

    def stest_multi_dim_in():
        pos_size = (2, 3, 2)
        a_size = (2, 4, 5)
        model = MLP(input_shape=pos_size, output_shape=a_size)

        pos_1 = to_tensor(numpy.random.rand(64, prod(pos_size[1:])), device="cpu")
        pos_2 = to_tensor(numpy.random.rand(64, prod(pos_size[1:])), device="cpu")
        print(model(pos_1, pos_2)[0].shape)

    def stest_multi_dim_out():
        pos_size = (10,)
        a_size = (2, 1)
        model = MLP(input_shape=pos_size, hidden_layers=(100,), output_shape=a_size)

        pos_1 = to_tensor(numpy.random.rand(64, *pos_size), device="cpu")
        res = model(pos_1)
        print(len(res), res[0].shape, res[1].shape)

    def stest_multi_dim_both():
        pos_size = (2, 3)
        a_size = (2, 4, 5)
        model = MLP(input_shape=pos_size, output_shape=a_size)

        pos_1 = to_tensor(numpy.random.rand(64, pos_size[0]), device="cpu")
        pos_2 = to_tensor(numpy.random.rand(64, pos_size[1]), device="cpu")
        res = model(pos_1, pos_2)
        print(len(res), res[0].shape, res[1].shape, res[2].shape)

    stest_single_dim()
    stest_hidden_dim()
    stest_multi_dim_both()
    stest_multi_dim_out()
