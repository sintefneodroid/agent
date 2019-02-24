#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
  r"""
      This criterion is a implementation of Focal Loss, which is proposed in
      Focal Loss for Dense Object Detection.

          Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

      The losses are averaged across observations for each mini batch.
      Args:
          alpha(1D Tensor, Variable) : the scalar factor for this criterion
          gamma(float, double) : gamma > 0; reduces the relative loss for well-classified examples (p > .5),
                                 putting more focus on hard, misclassified examples
          size_average(bool): size_average(bool): By default, the losses are averaged over observations for
          each mini batch.
                              However, if the field size_average is set to False, the losses are
                              instead summed for each mini batch.
  """

  def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
    super(FocalLoss, self).__init__()
    if alpha is None:
      self.alpha = Variable(torch.ones(class_num, 1))
    else:
      if isinstance(alpha, Variable):
        self.alpha = alpha
      else:
        self.alpha = Variable(alpha)
    self.gamma = gamma
    self.class_num = class_num
    self.size_average = size_average

  def forward(self, inputs, targets):
    N = inputs.size(0)
    C = inputs.size(1)
    P = F.softmax(inputs,0)

    class_mask = inputs.data.new(N, C).fill_(0)
    class_mask = Variable(class_mask)
    ids = targets.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)

    if inputs.is_cuda and not self.alpha.is_cuda:
      self.alpha = self.alpha.cuda()
    alpha = self.alpha[ids.data.view(-1)]

    probs = (P * class_mask).sum(1).view(-1, 1)

    log_p = probs.log()

    batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

    if self.size_average:
      loss = batch_loss.mean()
    else:
      loss = batch_loss.sum()
    return loss


if __name__ == "__main__":
  alpha = torch.rand(21, 1)
  focal_loss_func = FocalLoss(class_num=5, gamma=0)
  cross_entropy_func = nn.CrossEntropyLoss()
  N = 4
  C = 5
  inputs = torch.rand(N, C)
  targets = torch.LongTensor(N).random_(C)
  inputs_fl = Variable(inputs.clone(), requires_grad=True)
  targets_fl = Variable(targets.clone())

  inputs_ce = Variable(inputs.clone(), requires_grad=True)
  targets_ce = Variable(targets.clone())
  print('----inputs----')
  print(inputs)
  print('---target-----')
  print(targets)

  fl_loss = focal_loss_func(inputs_fl, targets_fl)
  ce_loss = cross_entropy_func(inputs_ce, targets_ce)
  print(f'ce = {ce_loss.item()}, fl ={fl_loss.item()}')
  fl_loss.backward()
  ce_loss.backward()
  print(inputs_ce.grad.data)
