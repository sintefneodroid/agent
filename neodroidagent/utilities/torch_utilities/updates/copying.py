#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
import torch


def copy_parameters(target: torch.nn.Module, source: torch.nn.Module) -> torch.nn.Module:
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(param.data)
  return target


def copy_state(*, target: torch.nn.Module, source: torch.nn.Module) -> torch.nn.Module:
  target.load_state_dict(source.state_dict())
  return target
