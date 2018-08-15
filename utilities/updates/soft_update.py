#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
import torch


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> torch.nn.Module:
  assert 0 <= tau <= 1
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(
        target_param.data * (1.0 - tau) + param.data * tau
        )

  return target
