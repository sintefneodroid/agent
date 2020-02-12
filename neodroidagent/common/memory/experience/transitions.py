#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Sized

from attr import dataclass

from neodroid.utilities.transformations.terminal_masking import (
    non_terminal_mask,
    non_terminal_numerical_mask,
)
from warg.mixins import IndexDictTuplesMixin, IterValuesMixin

__author__ = "Christian Heider Nielsen"

__all__ = [
    "Transition",
    "TransitionPoint",
    "SampleTransitionPoint",
    "DiscountedTransitionPoint",
]


@dataclass
class Transition(IterValuesMixin, IndexDictTuplesMixin):
    """
__slots__=['state','action','successor_state']
"""

    __slots__ = ["state", "action", "successor_state"]
    state: Any
    action: Any
    successor_state: Any

    @staticmethod
    def get_fields() -> Sized:
        """

    @return:
    """
        return Transition.__slots__

    def __len__(self):
        """

    @return:
    """
        return len(self.state)


@dataclass
class TransitionPoint(Transition):
    """
__slots__=['state','action','successor_state','signal,'terminal']
"""

    __slots__ = Transition.__slots__ + ["signal", "terminal"]
    state: Any
    action: Any
    successor_state: Any
    signal: Any
    terminal: Any

    def __post_init__(self):
        """

    @return:
    """
        if self.terminal:
            self.successor_state = None

    @staticmethod
    def get_fields() -> Sized:
        """

    @return:
    """
        return TransitionPoint.__slots__

    @property
    def non_terminal(self):
        """

    @return:
    """
        return non_terminal_mask(self.terminal)

    @property
    def non_terminal_numerical(self):
        """

    @return:
    """
        return non_terminal_numerical_mask(self.terminal)


@dataclass
class SampleTransitionPoint(TransitionPoint):
    """
__slots__=['state','action','successor_state','signal','terminal','distribution']
"""

    __slots__ = TransitionPoint.__slots__ + ["distribution"]
    state: Any
    action: Any
    successor_state: Any
    signal: Any
    terminal: Any
    distribution: Any

    @staticmethod
    def get_fields() -> Sized:
        """

    @return:
    """
        return SampleTransitionPoint.__slots__

    def __post_init__(self):
        pass
        # self.distribution = [self.distribution for _ in range(len(self.action))]


@dataclass
class DiscountedTransitionPoint(Transition):
    """
__slots__=['state'
,'action',
'successor_state',
'terminal',
'action_log_prob',
"discounted_return",
    ]
"""

    __slots__ = Transition.__slots__ + [
        "terminal",
        "action_log_prob",
        "discounted_return",
    ]
    state: Any
    action: Any
    successor_state: Any
    terminal: Any
    action_log_prob: Any
    discounted_return: Any

    @staticmethod
    def get_fields() -> Sized:
        """

    @return:
    """
        return DiscountedTransitionPoint.__slots__


if __name__ == "__main__":
    t = Transition(1, 2, 3)
    tp = TransitionPoint(*t, 4, 5)
    adtp = DiscountedTransitionPoint(*t, 6)

    print(t, tp, adtp)

    print([*adtp])
