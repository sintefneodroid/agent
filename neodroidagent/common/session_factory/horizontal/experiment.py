#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 19/01/2020
           """

import base64
import os
import pickle
import re
import time
from pathlib import Path

from cloudpickle import cloudpickle
from draugr.torch_utilities import CategoricalMLP

from neodroid.environments.droid_environment import DictUnityEnvironment
from neodroidagent.agents import SoftActorCriticAgent

_find_unsafe = re.compile(r"[a-zA-Z\d_^@%+=:,./-]").search


def _shellquote(s):
    """Return a shell-escaped version of the string *s*."""
    if not s:
        return "''"

    if _find_unsafe(s) is None:
        return s

    # use single quotes, and put single quotes into double quotes
    # the string $'b is then quoted as '$'"'"'b'

    return "'" + s.replace("'", "'\"'\"'") + "'"


def _to_param_val(v):
    if v is None:
        return ""
    elif isinstance(v, list):
        return " ".join(map(_shellquote, list(map(str, v))))
    else:
        return _shellquote(str(v))


def to_local_command(
    params, python_command="python", script="garage.experiment.experiment_wrapper"
):
    command = f"{python_command} -m {script}"

    garage_env = eval(os.environ.get("GARAGE_ENV", "{}"))
    for k, v in garage_env.items():
        command = f"{k}={v} " + command
    pre_commands = params.pop("pre_commands", None)
    post_commands = params.pop("post_commands", None)
    if pre_commands is not None or post_commands is not None:
        print(
            "Not executing the pre_commands: ",
            pre_commands,
            ", nor post_commands: ",
            post_commands,
        )

    for k, v in params.items():
        if isinstance(v, dict):
            for nk, nv in v.items():
                if str(nk) == "_name":
                    command += f"  --{k} {_to_param_val(nv)}"
                else:
                    command += f"  --{k}_{nk} {_to_param_val(nv)}"
        else:
            command += f"  --{k} {_to_param_val(v)}"
    return command


class Experiment:
    def __init__(self, log_dir="", save_dir="", render_environment=False):
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
                task["exp_name"] = f"{exp_prefix}_{time.time()}_{exp_count:04n}"

            if task.get("log_dir", None) is None:
                task["log_dir"] = (
                    f"{Path.cwd() / 'data'}/local/{exp_prefix.replace('_', '-')}/"
                    f"{task['exp_name']}"
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

    env = DictUnityEnvironment(env_name=ENV)

    policy = CategoricalMLP(
        input_shape=env.observation_space,
        output_shape=env.action_space,
        hidden_sizes=(32, 32),
    )

    agent = SoftActorCriticAgent(
        policy=policy, max_path_length=100, discount=0.99, max_kl_step=0.01
    )

    with Experiment(log_dir="", save_dir="", render_environment=False) as experiment:
        experiment.setup(agent, env)
        experiment.train(n_epochs=100, batch_size=4000)
