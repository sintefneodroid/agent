#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Christian Heider Nielsen"

import random
from collections import deque

import numpy

from neodroidagent.common.transitions.transitions import Transition
from warg.arguments import wrap_args

__all__ = ["ReplayBuffer", "ReplayBufferNumpy"]


class ReplayBuffer(object):
    def __init__(self, capacity=int(3e6)):
        self._buffer = deque(maxlen=capacity)

    def add(self, item):
        self._buffer.append(item)

    def sample(self, batch_size):
        assert batch_size <= len(self._buffer)
        return random.sample(self._buffer, batch_size)

    def __len__(self):
        return len(self._buffer)

    @wrap_args(Transition)
    def add_transition(self, transition):
        self.add(transition)

    def sample_transitions(self, num):
        values = self.sample(num)
        batch = Transition(*zip(*values))

        return batch


class ReplayBufferNumpy:
    def __init__(self, capacity=int(3e6)):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        o = map(numpy.stack, zip(*batch))
        return o

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    rb = ReplayBuffer()
    rbn = ReplayBufferNumpy()
