#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest

from neodroidagent.agents.torch_agents.model_free import ddpg_test
from neodroidagent.agents.torch_agents.model_free import pg_test
from neodroidagent.agents.torch_agents.model_free import dqn_test

__author__ = 'Christian Heider Nielsen'
__doc__ = ''


@pytest.mark.slow
def test_pg_agent():
  pg_test(1)


@pytest.mark.slow
def test_dqn_agent():
  dqn_test(1)


@pytest.mark.slow
def test_ppo_agent():
  pass
  # ppo_test(1)


@pytest.mark.slow
def test_ddpg_agent():
  ddpg_test(1)


if __name__ == '__main__':
  test_ddpg_agent()
  test_ppo_agent()
