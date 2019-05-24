#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import namedtuple

__author__ = 'cnheider'
__doc__ = ''

ExplorationSpecification = namedtuple('ExplorationSpecification',
                                      ['start',
                                       'end',
                                       'decay'])
