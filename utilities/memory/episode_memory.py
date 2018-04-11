class Trajectory:
  def __init__(self):
    self.signals = []
    self.log_probs = []
    self.entropies = []

  def remember(self, reward, log_prob, entropy):
    self.signals.append(reward)
    self.log_probs.append(log_prob)
    self.entropies.append(entropy)

  def forget(self):
    del self.signals[:]
    del self.log_probs[:]
    del self.entropies[:]

  def retrieve(self):
    return self.signals, self.log_probs, self.entropies


class ExpandableBuffer:
  def __init__(self):
    self.memory = []

  def remember(self, transition):
    self.memory.append(transition)

  def forget(self):
    del self.memory[:]
