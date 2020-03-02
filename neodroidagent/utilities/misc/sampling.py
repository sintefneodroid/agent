#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 10/02/2020
           """

from typing import Tuple

import torch
from torch.distributions import Distribution, Normal


def normal_tanh_reparameterised_sample(
    dis: Normal, epsilon=1e-6
) -> Tuple[torch.tensor, torch.tensor]:
    """
    The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution.
    The TanhNorm forces the Gaussian with infinite action range to be finite.

    For the three terms in this log-likelihood estimation:
     (1). the first term is the log probability of action as in common stochastic Gaussian action policy
     (without Tanh); \
    (2). the second term is the caused by the Tanh(), as shown in appendix C. Enforcing Action Bounds of
    https://arxiv.org/pdf/1801.01290.pdf, the epsilon is for preventing the negative cases in log


@param dis:
@param epsilon:
@return:
"""

    z = dis.rsample()  # for reparameterisation trick (mean + std * N(0,1))
    action = torch.tanh(z)
    log_prob = torch.sum(
        dis.log_prob(z) - torch.log(1 - action.pow(2) + epsilon), dim=-1, keepdim=True
    )
    return action, log_prob


if __name__ == "__main__":
    ob_shape = (10, 3)
    mean = torch.rand(ob_shape)
    std = torch.rand(ob_shape)
    dist = torch.distributions.Normal(mean, std)

    print(dist.rsample())
    print(normal_tanh_reparameterised_sample(dist))
