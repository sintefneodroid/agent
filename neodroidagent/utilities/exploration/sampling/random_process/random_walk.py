#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .random_process import RandomProcess

__author__ = "Christian Heider Nielsen"

import random


class RandomWalk(RandomProcess):
    def reset(self):
        pass

    def __init__(self, options=[-1, +1]):
        self.options = options

    def sample(self, size=1):
        return random.choice(self.options)


def main():
    random_process = RandomWalk()

    for i in range(1000):
        print(random_process.sample())


if __name__ == "__main__":
    main()
