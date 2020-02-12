#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from models import MLP

__author__ = "Christian Heider Nielsen"
__doc__ = ""

import torch
from torch import nn
from torch.nn import functional as F


class RecurrentCategoricalMLP(MLP):
    def __init__(self, r_hidden_layers=10, **kwargs):
        super().__init__(**kwargs)
        self._r_hidden_layers = r_hidden_layers
        self._r_input_shape = self._output_shape + r_hidden_layers

        self.hidden = nn.Linear(
            self._r_input_shape, r_hidden_layers, bias=self._use_bias
        )
        self.out = nn.Linear(self._r_input_shape, r_hidden_layers, bias=self._use_bias)

        self._prev_hidden_x = torch.zeros(r_hidden_layers)

    def forward(self, x, **kwargs):
        x = super().forward(x, **kwargs)
        combined = torch.cat((x, self._prev_hidden_x), 1)
        out_x = self.out(combined)
        hidden_x = self.hidden(combined)
        self._prev_hidden_x = hidden_x

        return F.softmax(out_x, dim=-1)


class ExposedRecurrentCategoricalMLP(RecurrentCategoricalMLP):
    def forward(self, x, hidden_x, **kwargs):
        self._prev_hidden_x = hidden_x
        out_x = super().forward(x, **kwargs)

        return F.softmax(out_x, dim=-1), self._prev_hidden_x
