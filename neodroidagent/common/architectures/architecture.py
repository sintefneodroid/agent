#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any, Sequence

import torch
from torch import nn

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""
__all__ = ["Architecture"]

from draugr.torch_utilities import get_num_parameters
from warg import drop_unused_kws
from draugr import indent_lines


class Architecture(nn.Module, ABC):
  """

"""

  @drop_unused_kws
  def __init__(self, input_shape: Sequence[int], output_shape: Sequence[int]):
    super().__init__()
    self._input_shape = input_shape
    self._output_shape = output_shape

  @property
  def input_shape(self) -> Sequence[int]:
    """

@return:
@rtype:
"""
    return self._input_shape

  @property
  def output_shape(self) -> Sequence[int]:
    """

@return:
@rtype:
"""
    return self._output_shape

  def sample_input(self)-> Any:
    return torch.empty(1, *self.input_shape, device="cpu")

  def __repr__(self):
    num_trainable_params = get_num_parameters(self, only_trainable=True)
    num_params = get_num_parameters(self, only_trainable=False)

    dict_repr = indent_lines(f"{self.__dict__}")

    trainable_params_str = indent_lines(
        f"trainable/num_params: {num_trainable_params}/{num_params}\n"
        )

    return f"{super().__repr__()}\n{dict_repr}\n{trainable_params_str}"


if __name__ == "__main__":
  a = Architecture()

  print(a)
