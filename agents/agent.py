class Agent(object):
  """
  All agent should inherit from this class
  """

  def __init__(self):
    pass

  def sample_action(self, state):
    raise NotImplementedError()

  def optimise_wrt(self, error):
    raise NotImplementedError()

  def rollout(self, init_obs, env):
    raise NotImplementedError()

  def infer(self, env, render=True):
    raise NotImplementedError()
