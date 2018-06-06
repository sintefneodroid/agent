from collections import namedtuple

OptimiserSpecification = namedtuple('OptimiserSpecification', ['constructor', 'kwargs'])

ConciseArchSpecification = namedtuple('ArchSpecification', ['input_size',
                                                     'hidden_layers',
                                                     'output_size',
                                                     'activation',
                                                     'use_bias'
                                                     ])
