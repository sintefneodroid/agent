#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import pytest

from agent.memory import ReplayBuffer, TrajectoryBuffer, TransitionBuffer

__author__ = 'cnheider'
__doc__ = ''


def test_replay_buffer():
  rb = ReplayBuffer()
  a = tuple(range(3))
  rb.add(a)
  b = rb.sample(1)
  assert [a] == b, f'Expected {a} and {b} to be equal'


def test_replay_buffer_more():
  rb = ReplayBuffer()
  a = tuple(range(3))
  rb.add(a)
  b = rb.sample(1)
  c = tuple(range(6))
  rb.add(c)
  d = rb.sample(2)
  assert d.__contains__(c)
  assert len(d) == 2
  assert [a] == b, f'Expected {a} and {b} to be equal'


def test_transition_buffer():
  rb = TransitionBuffer()
  a = tuple(range(3))
  rb._add(a)
  b = rb._sample(1)
  assert [a] == b, f'Expected {a} and {b} to be equal'


@pytest.mark.slow
def test_transition_buffer_list():
  rb = TransitionBuffer()
  a = numpy.random.random((9, 9))
  for e in a:
    rb.add_transition(e, None, None, None, None)
  b, *_ = rb.sample_transitions(9)
  assert numpy.array([a.__contains__(i) for i in b]).all(), f'Expected {a} to cover {b}'
  assert (a != b).all(), f'Expected {a[0]} and {b[0]} to be not equal'


def test_trajectory_buffer():
  rb = TrajectoryBuffer()
  a = tuple(range(3))
  rb.add_point(a, None, None)
  b, *_ = rb.retrieve_trajectory()
  assert (a,) == b, f'Expected {a} and {b} to be equal'
  rb.clear()
  c, *_ = rb.retrieve_trajectory()
  assert c is None, f'Expected {c} to be None'


def test_trajectory_buffer_more():
  rb = TrajectoryBuffer()
  a = tuple(range(3))
  rb.add_point(a, None, None)
  b, *_ = rb.retrieve_trajectory()
  c = tuple(range(6))
  rb.add_point(c, None, None)
  d, *_ = rb.retrieve_trajectory()
  assert d.__contains__(c)
  assert len(d) == 2
  assert (a,) == b, f'Expected {a} and {b} to be equal'


@pytest.mark.slow
def test_trajectory_list():
  rb = TrajectoryBuffer()
  a = numpy.random.random((9, 9))
  for e in a:
    rb.add_point(e, None, None)
  b, *_ = rb.retrieve_trajectory()
  assert (a == b).all(), f'Expected {a} and {b} to be equal'


if __name__ == '__main__':
  test_replay_buffer()
  test_replay_buffer_more()
  test_transition_buffer()
  test_trajectory_buffer()
  test_trajectory_buffer_more()
  test_trajectory_list()
  test_transition_buffer_list()
