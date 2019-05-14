#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
from neodroid.wrappers.utility_wrappers.action_encoding_wrappers import NeodroidWrapper

from agent.agents.ddpg_agent import DDPGAgent
from agent.agents.dqn_agent import DQNAgent
from agent.agents.pg_agent import PGAgent
from agent.agents.ppo_agent import PPOAgent
from samples.simple_environments import \
  StatefullEnvironment

__author__ = 'cnheider'
__doc__ = 'Tests of agents'


@pytest.mark.slow
def test_pg_agent():
  env = NeodroidWrapper(StatefullEnvironment())
  agent = PGAgent()
  # agent_test(agent,env)


@pytest.mark.slow
def test_dqn_agent():
  env = NeodroidWrapper(StatefullEnvironment())
  agent = DQNAgent()
  # agent_test(agent,env)


@pytest.mark.slow
def test_ppo_agent():
  env = NeodroidWrapper(StatefullEnvironment())
  agent = PPOAgent()
  # agent_test(agent,env)


@pytest.mark.slow
def test_ddpg_agent():
  env = NeodroidWrapper(StatefullEnvironment())
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
  test_pg_agent()
  test_dqn_agent()
  test_ppo_agent()
  test_ddpg_agent()
