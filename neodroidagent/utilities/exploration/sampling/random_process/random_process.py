#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC

__author__ = "Christian Heider Nielsen"


class RandomProcess(ABC):
    def __init__(self, **kwargs):
        pass

    def reset(self):
        raise NotImplementedError

    def sample(self, size):
        raise NotImplementedError
