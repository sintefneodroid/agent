#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy

from agent.utilities import to_tensor, torch

__author__ = 'cnheider'
__doc__ = ''


def test_to_tensor_none():
  try:
    tensor = to_tensor(None)
  except:
    return
  assert False


def test_to_tensor_empty_list():
  try:
    tensor = to_tensor([])
  except:
    return
  assert False


def test_to_tensor_empty_tuple():
  try:
    tensor = to_tensor(())
  except:
    return
  assert False


def test_to_tensor_list():
  ref = [0]
  tensor = to_tensor(ref)
  assert tensor.equal(torch.FloatTensor([0]))


def test_to_tensor_multi_list():
  ref = [[0], [1]]
  tensor = to_tensor(ref)
  assert tensor.equal(torch.FloatTensor([[0], [1]]))


def test_to_tensor_tuple():
  ref = (0,)
  tensor = to_tensor(ref)
  assert tensor.equal(torch.FloatTensor([0]))


def test_to_tensor_multi_tuple():
  ref = ([0], [1])
  tensor = to_tensor(ref)
  assert tensor.equal(torch.FloatTensor([[0], [1]]))


def test_to_tensor_from_numpy_tensor():
  ref = torch.from_numpy(numpy.random.sample((1, 2)))
  tensor = to_tensor(ref, dtype=torch.double)
  assert tensor.equal(ref)


def test_to_tensor_float_tensor():
  ref = torch.FloatTensor([0])
  tensor = to_tensor(ref)
  assert tensor.equal(ref)


if __name__ == '__main__':
  pass
