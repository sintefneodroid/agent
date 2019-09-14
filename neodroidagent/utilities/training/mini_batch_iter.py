#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sequence

from neodroidagent.utilities.specifications import AdvantageDiscountedTransition, ValuedTransition

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''
           '''


def mini_batch_iter(mini_batch_size: int,
                    batch: Sequence[ValuedTransition]) -> iter:
  batch_size = len(batch)
  for _ in range(batch_size // mini_batch_size):
    rand_ids = numpy.random.randint(0, batch_size, mini_batch_size)
    a = batch[:, rand_ids]
    yield ValuedTransition(*a)


import numpy

from torch.utils.data import Dataset


class AdvDisDataset(Dataset):
  """
   * ``N`` - number of parallel environments
   * ``T`` - number of time steps explored in environments

  Dataset that flattens ``N*T*...`` arrays into ``B*...`` (where ``B`` is equal to ``N*T``) and returns
  such rows
  one by one. So basically we loose information about sequence order and we return
  for example one state, action and reward per row.

  It can be used for ``Model``'s that does not need to keep the order of events like MLP models.

  For ``LSTM`` use another implementation that will slice the dataset differently
  """

  def __init__(self, arrays: Sequence[AdvantageDiscountedTransition]) -> None:
    super().__init__()
    vt = numpy.concatenate(*zip(*arrays), axis=1)
    self.arrays = vt

  def __getitem__(self, index) -> AdvantageDiscountedTransition:
    return self.arrays[:, index]

  def __len__(self):
    return len(self.arrays)
