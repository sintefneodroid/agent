from agents.agent import Agent


class ValueAgent(Agent):
  """
  All value iteration agents should inherit from this class
  """

  def __init__(self):
    super().__init__()

  def forward(self, state, *args, **kwargs):
    raise NotImplementedError()
