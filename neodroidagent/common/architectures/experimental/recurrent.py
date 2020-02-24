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


class RecurrentBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super().__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs
