#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from typing import Sequence
import torch
from torch import nn, seed
import numpy
from numpy import prod

from draugr import to_tensor, xavier_init, fan_in_init, torch_seed, constant_init
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
        input_shape: Sequence = None,
        hidden_layers: Sequence = None,
        hidden_layer_activation: callable = torch.relu,
        output_shape: Sequence = None,
        use_bias: bool = True,
        use_dropout: bool = False,
        dropout_prob: float = 0.2,
        auto_build_hidden_layers_if_none=True,
        input_multiplier: int = 32,
        layer_num_modulator=100,
        output_multiplier: int = 16,
        default_init: callable = xavier_init,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert input_shape is not None
        assert output_shape is not None

        self._input_shape = None
        self._output_shape = None

        self.infer_input_shape(input_shape)
        self.infer_output_shape(output_shape)

        self._hidden_layer_activation = hidden_layer_activation
        self._use_bias = use_bias

        if not hidden_layers and auto_build_hidden_layers_if_none:
            hidden_layers = self.construct_progressive_hidden_layers(
                self._input_shape,
                self._output_shape,
                input_multiplier,
                output_multiplier,
                layer_num_divisor=layer_num_modulator,
            )
        elif not isinstance(hidden_layers, Sequence):
            hidden_layers = (hidden_layers,)

        if use_dropout:
            AugLinearLayer = lambda *a, **b: nn.Sequential(
                nn.Linear(*a, **b), nn.Dropout(p=dropout_prob)
            )
        else:
            AugLinearLayer = nn.Linear

        self._hidden_layers = hidden_layers
        self.num_of_layer = len(self._hidden_layers)
        previous_layer_size = self._hidden_layers[0] * self._input_shape[0]

        for i in range(1, self._input_shape[0] + 1):
            setattr(
                self,
                f"_in{i}",
                AugLinearLayer(
                    self._input_shape[1], self._hidden_layers[0], bias=self._use_bias
                ),
            )

        if self.num_of_layer > 0:
            for i in range(2, self.num_of_layer + 1):
                setattr(
                    self,
                    f"_fc{i}",
                    AugLinearLayer(
                        previous_layer_size,
                        self._hidden_layers[i - 1],
                        bias=self._use_bias,
                    ),
                )
                previous_layer_size = self._hidden_layers[i - 1]

        for i in range(1, self._output_shape[0] + 1):
            setattr(
                self,
                f"_out{i}",
                nn.Linear(
                    previous_layer_size, self._output_shape[1], bias=self._use_bias
                ),
            )

        if default_init:
            default_init(self)

    @staticmethod
    def construct_progressive_hidden_layers(
        _input_shape,
        _output_shape,
        input_multiplier=32,
        output_multiplier=16,
        layer_num_divisor=100,
    ):
        h_first_size = int(sum(_input_shape) * input_multiplier)
        h_last_size = int(sum(_output_shape) * output_multiplier)

        h_middle_size = int(numpy.sqrt(h_first_size * h_last_size))

        # num_layers = math.ceil(h_middle_size / layer_num_divisor)

        hidden_layers = NOD(h_first_size, h_middle_size, h_last_size).as_list()

        return hidden_layers

    def infer_input_shape(self, input_shape):
        if isinstance(input_shape, Sequence):
            assert len(input_shape) > 0, f"Got length {len(input_shape)}"
            if len(input_shape) > 2:
                # self._input_shape = functools.reduce(operator.mul,input_shape)
                self._input_shape = input_shape[0], prod(input_shape[1:])
                logging.info(
                    f"Flattening input {input_shape} to flattened vectorised input shape {self._input_shape}"
                )
            elif len(input_shape) < 2:
                self._input_shape = (1, input_shape[0])
                logging.info(
                    f"Inflating input shape {input_shape} to vectorised input shape {self._input_shape}"
                )
            else:
                self._input_shape = input_shape
        elif isinstance(input_shape, int):
            self._input_shape = (1, input_shape)
            logging.info(
                f"Inflating input shape {input_shape} to vectorised input shape {self._input_shape}"
            )
        else:
            raise ValueError(f"Can not use {input_shape} as input shape")

    def infer_output_shape(self, output_shape):
        if isinstance(output_shape, Sequence):
            assert len(output_shape) > 0, f"Got length {len(output_shape)}"
            if len(output_shape) > 2:
                self._output_shape = output_shape[0], prod(output_shape[1:])
                logging.info(
                    f"Flattening output shape {output_shape} to flattened vectorised output shape {self._output_shape}"
                )
            elif len(output_shape) < 2:
                self._output_shape = (1, output_shape[0])
                logging.info(
                    f"Inflating output shape {output_shape} to vectorised output shape {self._output_shape}"
                )
            else:
                self._output_shape = output_shape
        elif isinstance(output_shape, int):
            self._output_shape = (1, output_shape)
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

        x_len = len(x)
        if x_len != self.input_shape[0]:
            raise ValueError(
                f"{self.input_shape[0]} input arguments expected, {len(x)} was supplied"
            )

        ins = []
        for i in range(1, self._input_shape[0] + 1):
            ins.append(
                self._hidden_layer_activation(getattr(self, f"_in{i}")(x[i - 1]))
            )

        val = torch.cat(ins, dim=-1)
        for i in range(2, self.num_of_layer + 1):
            val = self._hidden_layer_activation(getattr(self, f"_fc{i}")(val))

        outs = []
        for i in range(1, self._output_shape[0] + 1):
            outs.append(getattr(self, f"_out{i}")(val))

        return outs


if __name__ == "__main__":
    torch_seed(4)

    def stest_single_dim():
        pos_size = (4,)
        a_size = (1,)
        model = MLP(input_shape=pos_size, output_shape=a_size)

        pos_1 = to_tensor(numpy.random.rand(64, pos_size[0]), device="cpu")
        print(model(pos_1))

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
        print(model(pos_1))
        print(model2(pos_1))
        print(model3(pos_1))

    def stest_multi_dim():
        pos_size = (2, 3, 2)
        a_size = (2, 4, 5)
        model = MLP(input_shape=pos_size, output_shape=a_size)

        pos_1 = to_tensor(numpy.random.rand(64, prod(pos_size[1:])), device="cpu")
        pos_2 = to_tensor(numpy.random.rand(64, prod(pos_size[1:])), device="cpu")
        print(model(pos_1, pos_2))

    # stest_single_dim()
    stest_hidden_dim()
    # stest_multi_dim()
