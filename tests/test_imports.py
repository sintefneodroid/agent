#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 01/08/2020
           """


def test_import():
    import neodroidagent

    print(neodroidagent.__version__)


def test_import_agents():
    import neodroidagent.agents

    print(neodroidagent.agents.__author__)


def test_common():
    import neodroidagent.common

    print(neodroidagent.common.__author__)


def test_entry_points():
    import neodroidagent.entry_points

    print(neodroidagent.entry_points.__author__)


def test_configs():
    import neodroidagent.configs

    print(neodroidagent.configs.__author__)


def test_utilities():
    import neodroidagent.utilities

    print(neodroidagent.utilities.__author__)
