import numpy as np
from torch.nn import init


def fan_in_init(tensor):
  fanin = tensor.size(1)
  v = 1.0 / np.sqrt(fanin)
  init.uniform(tensor, -v, v)
