import math

from warg.named_ordered_dictionary import NOD


def index_of_max(x):
  m = max(x)
  return x.index(m)


class UCB1:
  def __init__(self, n_arms):
    self._counts = [0 for _ in range(n_arms)]
    self._values = [0.0 for _ in range(n_arms)]

  def select_arm(self):
    n_arms = len(self._counts)

    for arm in range(n_arms):
      if self._counts[arm] == 0:
        return arm

    ucb_values = [0.0 for _ in range(n_arms)]
    total_counts = sum(self._counts)

    for arm in range(n_arms):
      bonus = math.sqrt((2 * math.log(total_counts)) / float(self._counts[arm]))
      ucb_values[arm] = self._values[arm] + bonus

    return index_of_max(ucb_values)

  def update_belief(self, arm_index, signal):
    self._counts[arm_index] = arm_draws = self._counts[arm_index] + 1
    arm_draws_float = float(arm_draws)

    value = self._values[arm_index]
    new_value = ((arm_draws - 1) / arm_draws_float) * value + (1 / arm_draws_float) * signal
    self._values[arm_index] = new_value

  @property
  def counts(self):
    return self._counts

  @property
  def values(self):
    return self._values

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


  arms = [NormalDistributionArm(4.01, 2.0), NormalDistributionArm(4, 2.0), NormalDistributionArm(3.99, 2.0)]

  ucb1 = UCB1(len(arms))

  ucb1.train(arms)

  print(ucb1.counts)
  print(ucb1.values)
