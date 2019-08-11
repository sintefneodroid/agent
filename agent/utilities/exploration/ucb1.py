import math
import sys

from warg.named_ordered_dictionary import NOD


def index_of_max(x):
  m = max(x)
  return x.index(m)


class UCB1:
  def __init__(self, n_options):
    self._counts = [0 for _ in range(n_options)]
    self._values = [1 / n_options for _ in range(n_options)]

  def select_arm(self):
    n_options = len(self._counts)

    for option in range(n_options):
      if self._counts[option] == 0:
        return option

    ucb_values = [0.0 for _ in range(n_options)]
    total_counts = sum(self._counts)

    for option in range(n_options):
      bonus = math.sqrt((2 * math.log(total_counts)) / float(self._counts[option]))
      ucb_values[option] = self._values[option] + bonus

    return index_of_max(ucb_values)

  def update_belief(self, option_index, signal):
    self._counts[option_index] = options_counts_int = self._counts[option_index] + 1
    options_counts_float = float(options_counts_int)

    value = self._values[option_index]
    new_value = ((options_counts_float - 1) / options_counts_float) * value + (
        1 / options_counts_float) * signal
    self._values[option_index] = new_value

  @property
  def counts(self):
    return self._counts

  @property
  def values(self):
    return self._values

  @property
  def normalised_values(self):
    s = len(self._values)
    normed = [0] * s
    for i in range(s):
      normed[i] = self._values[i] / (sum(self._values) + sys.float_info.epsilon)
    return normed

  def train(self,
            arms,
            rollouts=1000) -> NOD:
    for t in range(rollouts):
      chosen_arm = self.select_arm()
      reward = arms[chosen_arm].draw()
      self.update_belief(chosen_arm, reward)
    return NOD()


if __name__ == '__main__':
  import random


  class NormalDistributionArm:
    def __init__(self, mu, sigma):
      self.mu = mu
      self.sigma = sigma

    def draw(self):
      return random.gauss(self.mu, self.sigma)


  arms = [NormalDistributionArm(4.01, 2.0),
          NormalDistributionArm(4, 2.0),
          NormalDistributionArm(3.99, 2.0)]

  ucb1 = UCB1(len(arms))

  ucb1.train(arms)

  print(ucb1.counts)
  print(ucb1.values)
