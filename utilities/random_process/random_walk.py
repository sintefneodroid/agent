import random

from neodroid.utilities.random_process import RandomProcess


class RandomWalk(RandomProcess):
  def __init__(self, options=[-1, +1]):
    self.options = options

  def sample(self):
    return random.choice(self.options)


def main():
  random_process = RandomWalk()

  for i in range(1000):
    print(random_process.sample())


if __name__ == '__main__':
  main()