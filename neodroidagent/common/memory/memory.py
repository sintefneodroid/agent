#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27/02/2020
           """

from abc import abstractmethod
from typing import Any, Iterable

__all__ = ["Memory"]


class Memory:
    r"""

  """

    @abstractmethod
    def _sample(self, num: int = None) -> Iterable:
        r"""

    @param num:
    @return:
    """
        raise NotImplementedError

    @abstractmethod
    def _add(self, value: Any) -> None:
        r"""

    @param num:
    @return:
    """
        raise NotImplementedError

    def sample(self, num: int = None) -> Iterable:
        r"""

    @param num:
    @return:
    """
        return self._sample(num)

    def add(self, value: Any) -> None:
        r"""

    @param value:
    @return:
    """
        return self._add(value)

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def capacity(self) -> int:
        raise NotImplementedError
