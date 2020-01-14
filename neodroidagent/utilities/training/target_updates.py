#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 10/01/2020
           """
import torch

__all__ = ["update_target"]


def update_target(
    *,
    target_model: torch.nn.Module,
    source_model: torch.nn.Module,
    copy_percentage: float = 3e-2
):
    assert 0.0 <= copy_percentage <= 1.0
    for target_param, param in zip(
        target_model.parameters(), source_model.parameters()
    ):
        target_param.data.copy_(
            copy_percentage * param.data + (1 - copy_percentage) * target_param.data
        )
