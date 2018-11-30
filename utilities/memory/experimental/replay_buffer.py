import random
from collections import deque, namedtuple

from .transition import Transition


class ReplayBuffer3(object):

  def __init__(self, capacity):
    self._buffer = deque(maxlen=capacity)

  def add(self, item):
    self._buffer.append(item)

  def sample(self, batch_size):
    assert batch_size <= len(self._buffer)
    return random.sample(self._buffer, batch_size)

  def __len__(self):
    return len(self._buffer)

  def add_transition(self, *args):
    self.add(Transition(*args))

  def sample_transitions(self, num):
    values = self.sample(num)
    batch = Transition(*zip(*values))
    return batch

