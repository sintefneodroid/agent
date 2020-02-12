#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 18/01/2020
           """

from neodroidagent.entry_points import session_entry_point


def test_single_agent_session():
    from neodroidagent.agents.random_agent import RandomAgent
    from neodroidagent.common.session_factory.single_agent.parallel import (
        ParallelSession,
    )
    from neodroidagent.common.session_factory.single_agent.procedures.rollout_inference import (
        RolloutInference,
    )

    session_entry_point(
        RandomAgent,
        {},
        session=ParallelSession(
            procedure=RolloutInference,
            environment_type=True,
            auto_reset_on_terminal_state=True,
        ),
    )


if __name__ == "__main__":
    test_single_agent_session()
