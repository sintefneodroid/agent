#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"

import json
from ast import literal_eval as make_tuple

import numpy
import torch
from skimage import color, transform

__all__ = [
    "compute_state",
    "extract_and_compute_state",
    "process_rigidbody_data",
    "spatial_displacement",
    "normalise_position",
    "gray_downscale",
]


def compute_state(observations, configuration):
    """

:param observations:
:param configuration:
:return StateTensor:
"""
    StateTensorType = configuration.STATE_TYPE
    return StateTensorType([observations])


def extract_and_compute_state(info, configuration):
    """

:param info:
:param configuration:
:return StateTensor:
"""
    # observations = spatial_displacement()
    observations = []

    observations += info.get_observer(b"Goal1Observer").get_position()
    observations += info.get_observer(b"HandObserver").get_position()

    # observations += info.get_observer(b'LowerArmRigidbody').get_position()
    observations += info.get_observer(b"LowerArmRigidbody").get_rotation()
    lower_arm_data = info.get_observer(b"LowerArmRigidbody").get_data()
    observations += process_rigidbody_data(lower_arm_data)

    # observations += info.get_observer(b'UpperArmRigidbody').get_position()
    observations += info.get_observer(b"UpperArmRigidbody").get_rotation()
    upper_arm_data = info.get_observer(b"UpperArmRigidbody").get_data()
    observations += process_rigidbody_data(upper_arm_data)

    StateTensorType = configuration.STATE_TYPE
    return StateTensorType([observations])


def process_rigidbody_data(data):
    output = []
    parsed = json.loads(data.getvalue())
    output += list(make_tuple(parsed["Velocity"]))
    output += list(make_tuple(parsed["AngularVelocity"]))

    return output


def spatial_displacement(pos1, pos2):
    """

:param pos1:
:param pos2:
:return:
"""
    return (numpy.array(pos1) - numpy.array(pos2)).flatten()


def normalise_position(elements, bounds):
    """

:param elements:
:param bounds:
:return:
"""
    normalised_0_1 = (numpy.array(elements) + numpy.array(bounds)) / (
        numpy.array(bounds) * 2
    )
    return normalised_0_1.flatten()


def gray_downscale(state, configuration):
    StateTensorType = configuration.StateTensorType
    gray_img = color.rgb2gray(state)
    downsized_img = transform.resize(gray_img, (84, 84), mode="constant")
    state = torch.from_numpy(downsized_img).type(StateTensorType)  # 2D image tensor
    return torch.stack([state], 0).unsqueeze(0)
