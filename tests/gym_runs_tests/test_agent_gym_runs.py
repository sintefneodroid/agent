#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest

__author__ = "Christian Heider Nielsen"
__doc__ = ""


@pytest.mark.slow
def test_pg_agent():
    pass
    # pg_test()


@pytest.mark.slow
def test_dqn_agent():
    pass
    # dqn_test()


@pytest.mark.slow
def test_ppo_agent():
    pass
    # ppo_test(1)


@pytest.mark.slow
def test_ddpg_agent():
    pass
    # ddpg_test(1)


if __name__ == "__main__":
    test_ddpg_agent()
    test_ppo_agent()
