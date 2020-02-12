#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import sys

from neodroidagent import PROJECT_APP_PATH

__author__ = "Christian Heider Nielsen"
__doc__ = r"""This script will open data the directory of Neodroid Agent platform"""


def main():
    print(
        f"Opening default data directory ({PROJECT_APP_PATH.user_data}) of the Neodroid Agent platform using "
        f"the default filemanager"
    )

    if sys.platform == "win32":
        subprocess.Popen(["start", PROJECT_APP_PATH.user_data], shell=True)

    elif sys.platform == "darwin":
        subprocess.Popen(["open", PROJECT_APP_PATH.user_data])

    else:
        # try:
        subprocess.Popen(["xdg-open", PROJECT_APP_PATH.user_data])
    # except OSError:


if __name__ == "__main__":
    main()
