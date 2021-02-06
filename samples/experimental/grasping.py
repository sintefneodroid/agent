#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 19-10-2020
           """

if __name__ == "__main__":

    def main():
        from pybullet_envs.bullet.kuka_diverse_object_gym_env import (
            KukaDiverseObjectEnv,
        )

        env = KukaDiverseObjectEnv(
            isDiscrete=True,
            renders=True,
            height=84,
            width=84,
            maxSteps=2000,
            isTest=True,
        )
        # Disable file caching to keep memory usage small
        env._p.setPhysicsEngineParameter(enableFileCaching=False)

    main()
