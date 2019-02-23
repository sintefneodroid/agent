import numpy

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warg

import torch
import torch.nn as nn
import torch.functional as f
import numpy as np


class DICELossMultiClass(nn.Module):

    def __init__(self):
        super(DICELossMultiClass, self).__init__()

    def forward(self, output, mask):
        num_classes = output.size(1)
        dice_eso = 0
        for i in range(num_classes):
            probs = torch.squeeze(output[:, i, :, :], 1)
            mask = torch.squeeze(mask[:, i, :, :], 1)

            num = probs * mask
            num = torch.sum(num, 2)
            num = torch.sum(num, 1)

            # print( num )

            den1 = probs * probs
            # print(den1.size())
            den1 = torch.sum(den1, 2)
            den1 = torch.sum(den1, 1)

            # print(den1.size())

            den2 = mask * mask
            # print(den2.size())
            den2 = torch.sum(den2, 2)
            den2 = torch.sum(den2, 1)

            # print(den2.size())
            eps = 0.0000001
            dice = 2 * ((num + eps) / (den1 + den2 + eps))
            # dice_eso = dice[:, 1:]
            dice_eso += dice

        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss


class DICELoss(nn.Module):

    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, output, mask):

        probs = torch.squeeze(output, 1)
        mask = torch.squeeze(mask, 1)

        intersection = probs * mask
        intersection = torch.sum(intersection, 2)
        intersection = torch.sum(intersection, 1)

        den1 = probs * probs
        den1 = torch.sum(den1, 2)
        den1 = torch.sum(den1, 1)

        den2 = mask * mask
        den2 = torch.sum(den2, 2)
        den2 = torch.sum(den2, 1)

        eps = 1e-8
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
        # dice_eso = dice[:, 1:]
        dice_eso = dice

        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss


def dice_loss(pred, target, smooth=1.):
  pred = pred.contiguous()
  target = target.contiguous()

  intersection = (pred * target).sum(dim=2).sum(dim=2)

  loss = (1 - (
      (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

  return loss.mean()


def calculate_loss(pred, target,*, original=None, bce_weight=0.6):
  if bce_weight == None:
    t = target.data.cpu().numpy()
    bce_weight = numpy.count_nonzero(t) / numpy.size(t)

  bce_weight = numpy.clip(bce_weight,0.,1.)

  binary_cross_entropy = F.binary_cross_entropy(pred, target)

  dice = dice_loss(pred, target)

  if original is not None:
    pass
  #  reconstruction_loss = ae_loss(pred,original)

  loss = binary_cross_entropy * bce_weight + dice * (1 - bce_weight)

  metrics=dict()

  metrics['bce'] = binary_cross_entropy.data.cpu().numpy() * target.size(0)
  metrics['dice'] = dice.data.cpu().numpy() * target.size(0)
  #metrics['ae'] = reconstruction_loss.data.cpu().numpy() * target.size(0)
  metrics['loss'] = loss.data.cpu().numpy() * target.size(0)

  return warg.NOD.dict_of(loss,metrics)
