#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'

import numpy as np
import torch
import torch.nn as nn
import warg

def jaccard_similarity_score(pred, target, *, epsilon=1e-10):
  pred_flat = pred.contiguous().view(-1)  # have to use contiguous since they may from a torch.view op
  target_flat = target.contiguous().view(-1)

  intersection = (pred_flat * target_flat).sum() + epsilon
  union = (target_flat ** 2).sum() + (pred_flat ** 2).sum() + epsilon

  dice_coefficient = intersection / (union - intersection)

  return dice_coefficient


def jaccard_loss(pred, target, *, epsilon=1e-10):
  return 1 - jaccard_similarity_score(pred, target, epsilon=epsilon)

if __name__ == '__main__':
  h = torch.FloatTensor(np.array([[0, 1], [1, 1]]))
  j = torch.FloatTensor(np.ones((2, 2)))
  x = jaccard_loss(h, j)
  print(x)
