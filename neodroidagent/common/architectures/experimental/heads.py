#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from models import MLP
from torch import nn

__author__ = "Christian Heider Nielsen"
__doc__ = ""


class MultiHeadedMLP(MLP):
    def __init__(self, *, heads_hidden_sizes=(32, 64), heads=(2, 1), **kwargs):
        super().__init__(**kwargs)

        assert len(heads_hidden_sizes) == len(heads)

        self._heads_hidden_sizes = heads_hidden_sizes
        self._heads = heads

        self.num_of_heads = len(self._heads)
        if self.num_of_heads > 0:
            for i in range(1, self.num_of_heads + 1):
                head_hidden = nn.Linear(
                    self._output_shape,
                    self._heads_hidden_sizes[i - 1],
                    bias=self._use_bias,
                )
                setattr(self, f"subhead{str(i)}_hidden", head_hidden)
                head = nn.Linear(
                    self._heads_hidden_sizes[i - 1],
                    self._heads[i - 1],
                    bias=self._use_bias,
                )
                setattr(self, f"subhead{str(i)}", head)
        else:
            raise ValueError("Number of heads must be >0")

    def forward(self, x, **kwargs):
        x = super().forward(x, **kwargs)

        output = []
        for i in range(1, self.num_of_heads + 1):
            head_hidden = getattr(self, f"subhead{str(i)}_hidden")
            x_s = head_hidden(x)
            head = getattr(self, f"subhead{str(i)}")
            sub_res = head(x_s)

            # if not isinstance(sub_res, list):
            #  sub_res = [sub_res]

            output.append(sub_res)

        return output
