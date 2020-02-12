#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 10/02/2020
           """

import torch
from torch.distributions import Distribution


def tanh_reparameterised_sample(dis: Distribution, epsilon=1e-6):
    """
          # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # For the three terms in this log-likelihood estimation: \
        # (1). the first term is the log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        # (2). the second term is the caused by the Tanh(), \
        # as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        # (3). the third term is caused by the action range I used in this code is not (-1, 1) but with \
        # an arbitrary action range, which is slightly different from original paper.

        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.


  @param dis:
  @param epsilon:
  @return:
  """
    z = dis.rsample()  # for reparameterisation trick (mean + std * N(0,1))
    action = torch.tanh(z)
    log_prob = (dis.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)).sum(
        -1, keepdim=True
    )
    return action, log_prob
