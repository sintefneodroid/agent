#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
__doc__ = ''

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0),
              (0.0, 1.0),
              (1.0, 0.0),
              (1.0, 1.0)]

xor_outputs = [(0.0,),
               (1.0,),
               (1.0,),
               (0.0,)]


def eval_genomes(model):
  for xor_true_i, xor_true_o in zip(xor_inputs, xor_outputs):
    output = model.predict(xor_true_i)
    error = (output[0] - xor_true_o[0]) ** 2
    return error
