import random
from collections import deque

from agent.utilities.memory import Transition


class ReplayBuffer(object):

  def __init__(self, capacity=int(3e6)):
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
