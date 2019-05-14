import os.path as osp
import tempfile

import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import filelock


class MnistEnv(Env):
  def __init__(self,
               seed=0,
               episode_len=None,
               no_images=None
               ):

    from tensorflow.examples.tutorials.mnist import input_data
    # we could use temporary directory for this with a context manager and
    # TemporaryDirectory, but then each test that uses MNIST would re-download the data
    # this way the data is not cleaned up, but we only download it once per machine
    mnist_path = osp.join(tempfile.gettempdir(), 'MNIST_data')
    with filelock.FileLock(mnist_path + '.lock'):
      self.mnist = input_data.read_data_sets(mnist_path)

    self.np_random = np.random.RandomState()
    self.np_random.seed(seed)

    self.observation_space = Box(low=0.0, high=1.0, shape=(28, 28, 1))
    self.action_space = Discrete(10)
    self.episode_len = episode_len
    self.time = 0
    self.no_images = no_images

    self.train_mode()
    self.reset()

  def reset(self):
    self._choose_next_state()
    self.time = 0

    return self.state[0]

  def step(self, actions):
    signal = self._get_reward(actions)
    self._choose_next_state()
    terminal = False
    if self.episode_len and self.time >= self.episode_len:
      signal = 0
      terminal = True

    return self.state[0], signal, terminal, {}

  def train_mode(self):
    self.dataset = self.mnist.train

  def test_mode(self):
    self.dataset = self.mnist.test

  def _choose_next_state(self):
    max_index = (self.no_images if self.no_images is not None else self.dataset.num_examples) - 1
    index = self.np_random.randint(0, max_index)
    image = self.dataset.images[index].reshape(28, 28, 1) * 255
    label = self.dataset.labels[index]
    self.state = (image, label)
    self.time += 1

  def _get_reward(self, actions):
    return 1 if self.state[1] == actions else 0
