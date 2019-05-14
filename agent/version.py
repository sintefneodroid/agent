#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import os
from warnings import warn

__author__ = "cnheider"
__version__ = "0.1.0"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

RELEASE = False
DEBUG = False


def get_version(append_time=False):
  version = __version__
  if not version:
    version = os.getenv("VERSION", "0.0.0")

  if append_time or not RELEASE:
    now = datetime.datetime.utcnow()
    date_version = now.strftime("%Y%m%d%H%M%S")
    # date_version = time.time()

    if version:
      # Most git tags are prefixed with 'v' (example: v1.2.3) this is
      # never desirable for artifact repositories, so we strip the
      # leading 'v' if it's present.
      version = version[1:] if isinstance(version, str) and version.startswith("v") else version
    else:
      # Default version is an ISO8601 compliant datetime. PyPI doesn't allow
      # the colon ':' character in its versions, and time is required to allow
      # for multiple publications to master in one day. This datetime string
      # uses the 'basic' ISO8601 format for both its date and time components
      # to avoid issues with the colon character (ISO requires that date and
      # time components of a date-time string must be uniformly basic or
      # extended, which is why the date component does not have dashes.
      #
      # Publications using datetime versions should only be made from master
      # to represent the HEAD moving forward.
      warn(f"Environment variable VERSION is not set, only using datetime: {date_version}")

      # warn(f'Environment variable VERSION is not set, only using timestamp: {version}')

    version = f"{version}.{date_version}"

  return version


if __version__ is None:
  __version__ = get_version()


@property
def debug():
  return DEBUG
