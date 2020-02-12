#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import namedtuple

__author__ = "Christian Heider Nielsen"

__all__ = ["TrainingResume", "TR"]

TrainingResume = namedtuple("TrainingResume", ("models", "stats"))
TR = TrainingResume
