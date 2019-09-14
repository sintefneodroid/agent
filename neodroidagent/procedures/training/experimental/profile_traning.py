#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''
           '''

x = torch.randn((1, 1), requires_grad=True)
with torch.autograd.profiler.profile() as prof:
  for _ in range(100):
    y = x ** 2
    y.backward()
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
