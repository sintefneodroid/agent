#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from typing import List, Sequence

import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal

from draugr import fan_in_init
from neodroidagent.common.architectures.mlp import MLP

__author__ = "Christian Heider Nielsen"
__doc__ = ""
__all__ = [
    "ShallowStdNormalMLP",
    "MultiDimensionalNormalMLP",
    "MultiVariateNormalMLP",
    "MultipleNormalMLP",
]

from warg import passes_kws_to


class ShallowStdNormalMLP(MLP):
    def __init__(
        self,
        output_shape: Sequence = (2,),
        mean_head_activation: callable = None,
        **kwargs
    ):
        super().__init__(
            output_shape=output_shape,
            mean_head_activation=mean_head_activation,
            **kwargs
        )

        self.mean_head_activation = mean_head_activation
        self.log_std = nn.Parameter(torch.zeros(output_shape[-1]), requires_grad=True)

    def forward(
        self, *x, min_std=-20, max_std=2, **kwargs
    ) -> torch.distributions.Distribution:
        mean = super().forward(*x, min_std=min_std, **kwargs)
        if self.mean_head_activation:
            mean = self.mean_head_activation(mean)

        log_std = torch.clamp(self.log_std, min_std, max_std)
        std = log_std.exp().expand_as(mean)
        return Normal(mean, std)


class ShallowStdMultiVariateNormalMLP(MLP):
    def __init__(
        self,
        output_shape: Sequence = (2,),
        mean_head_activation: callable = None,
        **kwargs
    ):
        super().__init__(
            output_shape=output_shape,
            mean_head_activation=mean_head_activation,
            **kwargs
        )

        self.mean_head_activation = mean_head_activation
        self.log_std = nn.Parameter(torch.zeros(*output_shape), requires_grad=True)

    def forward(self, *x, min_std=-20, max_std=2, **kwargs):
        mean = super().forward(*x, min_std=min_std, **kwargs)
        if self.mean_head_activation:
            mean = self.mean_head_activation(mean)

        log_std = torch.clamp(self.log_std, min_std, max_std)
        std = log_std.exp().expand_as(mean)
        std = torch.diag_embed(std, 0, dim1=-2, dim2=-1)
        return MultivariateNormal(mean, std)


class MultiDimensionalNormalMLP(MLP):
    def __init__(
        self,
        output_shape: Sequence = (2,),
        mean_head_activation: callable = None,
        **kwargs
    ):
        output_shape = (*output_shape, *output_shape)
        assert len(output_shape) == 2

        self.mean_head_activation = mean_head_activation
        super().__init__(
            output_shape=output_shape,
            mean_head_activation=mean_head_activation,
            **kwargs
        )

    def forward(self, *x, min_std=-20, max_std=2, **kwargs) -> Normal:
        mean, log_std = super().forward(*x, min_std=min_std, **kwargs)
        if self.mean_head_activation:
            mean = self.mean_head_activation(mean)

        return Normal(mean, torch.clamp(log_std, min_std, max_std).exp())


class MultiVariateNormalMLP(MLP):
    @passes_kws_to(MLP.__init__)
    def __init__(
        self,
        output_shape: Sequence = (2,),
        mean_head_activation: callable = None,
        **kwargs
    ):
        output_shape = (*output_shape, *output_shape)
        assert len(output_shape) == 2

        self.mean_head_activation = mean_head_activation
        super().__init__(output_shape=output_shape, **kwargs)

    @passes_kws_to(MLP.forward)
    def forward(self, *x, min_std=-20, max_std=2, **kwargs) -> MultivariateNormal:
        mean, log_std = super().forward(*x, min_std=min_std, **kwargs)
        if self.mean_head_activation:
            mean = self.mean_head_activation(mean)

        log_std = torch.clamp(log_std, min_std, max_std)
        std = log_std.exp()
        std = torch.diag_embed(std, 0, dim1=-2, dim2=-1)
        return MultivariateNormal(mean, std)


class MultipleNormalMLP(MLP):
    def __init__(
        self, output_shape: int = 2, mean_head_activation: callable = None, **kwargs
    ):
        output_shape = (2,) * output_shape
        super().__init__(output_shape=output_shape, **kwargs)

        self.mean_head_activation = mean_head_activation
        fan_in_init(self)

    def forward(self, *x, min_std=-20, max_std=2, **kwargs) -> List[Normal]:
        out = super().forward(*x, min_std=min_std, **kwargs)
        outs = []
        for a in out:
            mean, log_std = a
            if self.mean_head_activation:
                mean = self.mean_head_activation(mean)
            outs.append(Normal(mean, torch.clamp(log_std, min_std, max_std).exp()))

        return outs


if __name__ == "__main__":

    def stest_normal():
        s = (10,)
        a = 10
        model = MultipleNormalMLP(input_shape=s, output_shape=a)

        inp = torch.rand(s)
        s_ = time.time()
        dis = model.forward(inp)
        print(dis)
        a_ = [d.sample() for d in dis]
        print(time.time() - s_, a_)

    def stest_multi_dim_normal():
        s = (4,)
        a = (10,)
        model = MultiDimensionalNormalMLP(input_shape=s, output_shape=a)

        inp = torch.rand(s)
        s_ = time.time()
        dis = model.forward(inp)
        print(dis)
        a_ = dis.sample()
        print(time.time() - s_, a_)

    def stest_multi_var_normal():
        s = (10,)
        a = (10,)
        model = MultiVariateNormalMLP(input_shape=s, output_shape=a)

        inp = torch.rand(s)
        s_ = time.time()
        dis = model.forward(inp)
        print(dis)
        a_ = dis.sample()
        print(time.time() - s_, a_)

    def stest_shallow():
        s = (10,)
        a = (10,)
        model = ShallowStdNormalMLP(input_shape=s, output_shape=a)

        inp = torch.rand(s)
        s_ = time.time()
        dis = model.forward(inp)
        print(dis)
        a_ = dis.sample()
        print(time.time() - s_, a_)

    stest_normal()
    print("\n")
    stest_multi_dim_normal()
    print("\n")
    stest_multi_var_normal()
    print("\n")
    stest_shallow()
