#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from experiments.segmentation.losses.dice_loss import dice_loss
from experiments.segmentation.losses.jaccard_loss import jaccard_loss

__author__ = 'cnheider'

import torch
import warg


def calculate_accum_loss(pred,
                         target,
                         reconstruction,
                         original):
  term_weight = 1 / 4

  seg_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target).mean()
  ae_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(reconstruction, original).mean()

  pred_soft = torch.sigmoid(pred)
  dice = dice_loss(pred_soft, target, epsilon=1).mean()
  jaccard = jaccard_loss(pred_soft, target, epsilon=1).mean()

  loss = seg_bce_loss * term_weight

  for term in [dice, jaccard, ae_bce_loss]:
    loss += term * term_weight

  n = target.size(0)

  bce_l = seg_bce_loss.data.cpu().numpy() * n
  dice_l = dice.data.cpu().numpy() * n
  jaccard_l = dice.data.cpu().numpy() * n
  ae_l = ae_bce_loss.data.cpu().numpy() * n
  total_l = loss.data.cpu().numpy() * n

  metrics = warg.NOD.dict_of(bce_l, dice_l, jaccard_l, ae_l, total_l)

  return warg.NOD.dict_of(loss, metrics)
