#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 02/01/2020
           """

__all__ = ["is_set_mod_zero_ret_alt"]


def is_set_mod_zero_ret_alt(mod, counter, *, ret=True, alt=None):
    """

  test if mod is set
  then test if counter % mod is 0
  if both tests are true return ret
  else return alt


  @param mod:
  @param counter:
  @param ret:
  @param alt:
  @return:
  """
    return ret if (mod and (counter % mod == 0)) else alt
