#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"

import numpy

import neodroid


def calculate_difficulty(current_difficulty, signal, steps_taken, configuration):
    # current_difficulty = numpy.random.normal(0,0.2,size=3) + current_difficulty

    if signal < -0.6:
        return current_difficulty  # * 0.8
    elif signal > 0.6 and steps_taken % configuration.UPDATE_DIFFICULTY_INTERVAL == 0:
        return min(current_difficulty * 1.4, 10)
    else:
        return current_difficulty


def sample_configuration2(current_difficulty, info):
    perturbation = numpy.random.normal(0, 0.2, size=3) * current_difficulty
    goal_pos = info.get_observer(b"GoalObserver").get_position()

    b = [
        neodroid.Configuration(
            "ManipulatorTriDOFConfigurableX", goal_pos[0] + perturbation[0]
        ),
        neodroid.Configuration(
            "ManipulatorTriDOFConfigurableY", goal_pos[1] + perturbation[1]
        ),
        neodroid.Configuration(
            "ManipulatorTriDOFConfigurableZ", goal_pos[2] + perturbation[2]
        ),
    ]

    return (
        b
        + color_sample("Wall1")
        + color_sample("Wall2")
        + color_sample("Wall3")
        + color_sample("Wall4")
        + color_sample("Roof")
        + color_sample("Ground")
    )


def sample_configuration(current_difficulty, info):
    perturbation = numpy.random.normal(0, 0.2, size=3) * current_difficulty
    x, y, z = info.get_observer(b"GoalObserver").get_position()

    b = [
        neodroid.Configuration("GoalTransformX", numpy.round(x + perturbation[0])),
        neodroid.Configuration("GoalTransformY", numpy.round(y + perturbation[1])),
        neodroid.Configuration("GoalTransformZ", numpy.round(z + perturbation[2])),
    ]

    return b


def color_sample(identifier):
    return [
        neodroid.Configuration(
            identifier + "ColorConfigurableR", numpy.random.sample()
        ),
        neodroid.Configuration(
            identifier + "ColorConfigurableG", numpy.random.sample()
        ),
        neodroid.Configuration(identifier + "ColorConfigurableB", numpy.random.sample())
        # neodroid.Configuration(id+'ColorConfigurableA',
        #                       numpy.random.sample()),
    ]
