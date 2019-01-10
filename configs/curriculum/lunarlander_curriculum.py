#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from warg import NOD

__author__ = 'cnheider'

CURRICULUM = NOD(**{
  'level1':{
    'when_reward':  0.5,
    'configurables':{
      'WallColorVariation':[0.0, 0.0, 0.0], 'StartBoundaryRadius':1
      },
    },
  'level2':{
    'when_reward':  0.7,
    'configurables':{
      'WallColorVariation':[0.1, 0.1, 0.1], 'StartBoundaryRadius':2
      },
    },
  'level3':{
    'when_reward':  0.8,
    'configurables':{
      'WallColorVariation':[0.5, 0.5, 0.5], 'StartBoundaryRadius':3
      },
    },
  })

CURRICULUM2 = NOD(**{
  'level1':{'when_reward':0.95, 'configurables':{'Difficulty':1}},
  'level2':{'when_reward':0.95, 'configurables':{'Difficulty':2}},
  'level3':{'when_reward':0.95, 'configurables':{'Difficulty':3}},
  })
