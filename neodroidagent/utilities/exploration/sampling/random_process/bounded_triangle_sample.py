#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

import numpy


def bounded_triangle_sample(a_set, mean=0.5, number=1):
    l = len(a_set)
    a = numpy.random.triangular(0, l * mean, l, number)
    a = int(numpy.floor(a)[0])

    return a_set[a]


if __name__ == "__main__":
    print(bounded_triangle_sample(numpy.arange(0, 10)))
