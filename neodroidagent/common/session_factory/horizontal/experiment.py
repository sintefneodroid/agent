#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 19/01/2020
           """

import base64
import os
import pickle

from cloudpickle import cloudpickle
from garage.experiment import to_local_command

from neodroid.environments.unity_environment import UnityEnvironment
from neodroidagent.agents import SACAgent, Agent
from neodroidagent.agents.numpy_agents.baseline_agent import LinearFeatureBaselineAgent
from neodroidagent.common import CategoricalMLP


class Experiment:
    def __init__(self, log_dir="", save_dir="", render=False):
        pass

    def __enter__(self):
        """Set self.sess as the default session.

    Returns:
        This local runner.

    """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave session."""

    def run_experiment(
        self,
        method_call=None,
        batch_tasks=None,
        exp_prefix="experiment",
        exp_name=None,
        log_dir=None,
        script="garage.experiment.experiment_wrapper",
        python_command="python",
        dry=False,
        env=None,
        variant=None,
        force_cpu=False,
        pre_commands=None,
        **kwargs,
    ):
        """Serialize the method call and run the experiment using the
    specified mode.

    Args:
        method_call (callable): A method call.
        batch_tasks (list[dict]): A batch of method calls.
        exp_prefix (str): Name prefix for the experiment.
        exp_name (str): Name of the experiment.
        log_dir (str): Log directory for the experiment.
        script (str): The name of the entrance point python script.
        python_command (str): Python command to run the experiment.
        dry (bool): Whether to do a dry-run, which only prints the
            commands without executing them.
        env (dict): Extra environment variables.
        variant (dict): If provided, should be a dictionary of parameters.
        force_cpu (bool): Whether to set all GPU devices invisible
            to force use CPU.
        pre_commands (str): Pre commands to run the experiment.

    """
        if method_call is None and batch_tasks is None:
            raise Exception("Must provide at least either method_call or batch_tasks")

        for task in batch_tasks or [method_call]:
            if not hasattr(task, "__call__"):
                raise ValueError("batch_tasks should be callable")
            # ensure variant exists
            if variant is None:
                variant = dict()

        if batch_tasks is None:
            batch_tasks = [
                dict(
                    kwargs,
                    pre_commands=pre_commands,
                    method_call=method_call,
                    exp_name=exp_name,
                    log_dir=log_dir,
                    env=env,
                    variant=variant,
                )
            ]

        global exp_count

        if force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        for task in batch_tasks:
            call = task.pop("method_call")
            data = base64.b64encode(cloudpickle.dumps(call)).decode("utf-8")
            task["args_data"] = data
            exp_count += 1

            if task.get("exp_name", None) is None:
                task["exp_name"] = f"{exp_prefix}_{timestamp}_{exp_count:04n}"

            if task.get("log_dir", None) is None:
                task["log_dir"] = "{log_dir}/local/{exp_prefix}/{exp_name}".format(
                    log_dir=osp.join(os.getcwd(), "data"),
                    exp_prefix=exp_prefix.replace("_", "-"),
                    exp_name=task["exp_name"],
                )

            if task.get("variant", None) is not None:
                variant = task.pop("variant")
                if "exp_name" not in variant:
                    variant["exp_name"] = task["exp_name"]
                task["variant_data"] = base64.b64encode(pickle.dumps(variant)).decode(
                    "utf-8"
                )
            elif "variant" in task:
                del task["variant"]
            task["env"] = task.get("env", dict()) or dict()
            task["env"]["GARAGE_FORCE_CPU"] = str(force_cpu)

        for task in batch_tasks:
            env = task.pop("env", None)
            command = to_local_command(
                task, python_command=python_command, script=script
            )
            print(command)
            if dry:
                return
            try:
                if env is None:
                    env = dict()
                os.subprocess.run(
                    command, shell=True, env=dict(os.environ, **env), check=True
                )
            except Exception as e:
                print(e)
                raise


if __name__ == "__main__":

    ENV = ""

    env = UnityEnvironment(env_name=ENV)

    policy = CategoricalMLP(
        input_shape=env.observation_space,
        output_shape=env.action_space,
        hidden_sizes=(32, 32),
    )

    agent = SACAgent(
        policy=policy, max_path_length=100, discount=0.99, max_kl_step=0.01
    )

    with Experiment(log_dir="", save_dir="", render=False) as experiment:
        experiment.setup(agent, env)
        experiment.train(n_epochs=100, batch_size=4000)
