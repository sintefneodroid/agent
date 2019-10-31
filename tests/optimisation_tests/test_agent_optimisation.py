#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest

from neodroid.environments.gym_environment import NeodroidGymWrapper
from neodroidagent.agents.torch_agents.model_free.actor_critic import PPOAgent, DDPGAgent
from neodroidagent.agents.torch_agents.model_free.off_policy.dqn_agent import DQNAgent
from neodroidagent.agents.torch_agents.model_free.on_policy.pg_agent import PGAgent

__author__ = 'Christian Heider Nielsen'
__doc__ = 'Tests of agents'


@pytest.mark.slow
def test_pg_agent_optimisation():
  env = NeodroidGymWrapper(None)
  agent = PGAgent()
  # agent_test(agent,env)


@pytest.mark.slow
def test_dqn_agent_optimisation():
  env = NeodroidGymWrapper(None)
  agent = DQNAgent()
  # agent_test(agent,env)


@pytest.mark.slow
def test_ppo_agent_optimisation():
  env = NeodroidGymWrapper(None)
  agent = PPOAgent()
  # agent_test(agent,env)


@pytest.mark.slow
def test_ddpg_agent_optimisation():
  env = NeodroidGymWrapper(None)
  agent = DDPGAgent()
  # agent_test(agent, env)


def agent_test(agent, env):
  agent.build(env)

  episode_signals = []
  for i in range(1000):
    state = env.reset()
    episode_signal, *_ = agent.rollout(state, env, max_length=30, train=True)
    if i > 1000 - 100:
      episode_signals.append(episode_signal)

  last_avg = sum(episode_signals) / 100
  assert last_avg > 10, f'Had {last_avg}'


if __name__ == '__main__':
  test_pg_agent_optimisation()
  test_dqn_agent_optimisation()
  test_ppo_agent_optimisation()
  test_ddpg_agent_optimisation()
