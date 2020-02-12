#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 10/01/2020
           """

import torch

__all__ = ["update_target", "hard_copy_params", "soft_copy_params"]

from draugr import fan_in_init


def update_target(
    *,
    target_model: torch.nn.Module,
    source_model: torch.nn.Module,
    copy_percentage: float = 1.0
):
    assert 0.0 <= copy_percentage <= 1.0
    if copy_percentage == 1.0:
        hard_copy_params(target_model, source_model)
    else:
        soft_copy_params(target_model, source_model, copy_percentage)


def hard_copy_params(target_model, source_model) -> None:
    for target_param, param in zip(
        target_model.parameters(), source_model.parameters()
    ):
        target_param.data.copy_(param.data)


def soft_copy_params(target_model, source_model, copy_percentage) -> None:
    for target_param, param in zip(
        target_model.parameters(), source_model.parameters()
    ):
        target_param.data.copy_(
            copy_percentage * param.data + (1 - copy_percentage) * target_param.data
        )


def inplace_polyak_update_params(target_model, source_model, copy_percentage) -> None:
    """
    update target networks by polyak averaging

    @param target_model:
    @param source_model:
    @param copy_percentage:
    @return:
    """
    with torch.no_grad():
        for p, p_targ in zip(source_model.parameters(), target_model.parameters()):
            p_targ.data.mul_(copy_percentage)
            p_targ.data.add_((1 - copy_percentage) * p.data)


if __name__ == "__main__":
    a = torch.nn.Linear(3, 4)
    b = torch.nn.Linear(3, 4)
    fan_in_init(b)
    # assert a.weight.eq(b.weight).all()
    update_target(target_model=b, source_model=a)
    assert a.weight.eq(b.weight).all()

    fan_in_init(b)
    update_target(target_model=a, source_model=b, copy_percentage=0.5)
    assert not a.weight.eq(b.weight).all()
