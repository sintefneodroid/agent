
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'

from collections import namedtuple


ConciseArchSpecification = namedtuple('ConciseArchSpecification', ['input_size',
                                                                   'hidden_layers',
                                                                   'output_size',
                                                                   'activation',
                                                                   'use_bias'
                                                                   ])