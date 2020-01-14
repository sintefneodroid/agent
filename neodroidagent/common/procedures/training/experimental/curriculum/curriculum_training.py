#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from pathlib import Path

import torch
from neodroidagent.agents.torch_agents.model_free import PGAgent
from tqdm import tqdm

import neodroidagent.configs.curriculum.curriculum_config as C
from draugr.visualisation import sprint
from draugr.writers import TensorBoardPytorchWriter, Union
from neodroid.wrappers import NeodroidCurriculumWrapper
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.utilities.exploration import sample
from neodroidagent.utilities.specifications import TR
from samples.rl import (
    display_actor_configurations,
    estimate_entire_state_space,
    estimate_initial_state_expected_return,
    get_initial_configuration_from_goal,
)
from warg.arguments import get_upper_case_vars_or_protected_of

__author__ = "Christian Heider Nielsen"

tqdm.monitor_interval = 0
torch.manual_seed(C.SEED)
# neo.seed(C.SEED)

"""stats = draugr.StatisticCollection(
    stats={
      'signals',
      'lengths',
      'entropies',
      'value_estimates',
      'sample_lengths'},
    measures={
      'variance',
      'mean'})
"""

_keep_stats = False
_plot_stats = False
_keep_seed_if_not_replaced = False

# _random_process = OrnsteinUhlenbeckProcess(0.5, size=_environment.action_space.shape[0])
_random_process = None


def save_snapshot():
    _agent.save(C)
    # stats.save(**configs.to_dict(C))


def main(
    agent: TorchAgent,
    environment: NeodroidCurriculumWrapper,
    *,
    log_directory: Union[str, Path],
    rollouts: int = 1000,
    render_frequency: int = 100,
    stat_frequency: int = 10,
    disable_stdout: bool = False,
    full_state_evaluation_frequency=20,
    **kwargs,
) -> TR:
    assert isinstance(environment, NeodroidCurriculumWrapper)

    with torch.autograd.detect_anomaly():
        with TensorBoardPytorchWriter(str(log_directory)) as metric_writer:
            _episode_i = 0
            _step_i = 0

            random_motion_length = C.RANDOM_MOTION_HORIZON
            training_start_timestamp = time.time()

            initial_configuration = get_initial_configuration_from_goal(environment)
            print("Generating initial state from goal configuration")
            S_prev = environment.generate_trajectory_from_configuration(
                initial_configuration,
                random_motion_length,
                random_process=_random_process,
            )

            train_session = range(1, rollouts + 1)
            train_session = tqdm(train_session, leave=False, disable=False)

            for i in train_session:
                if not environment.is_connected:
                    break

                S_initial = []
                S_candidate = []

                reached_dead_end = True

                if i % full_state_evaluation_frequency == 0:
                    print("Estimating entire state space")
                    estimate_entire_state_space(
                        environment,
                        agent,
                        C,
                        # statistics=None,
                        save_snapshot=save_snapshot,
                    )

                num_candidates = tqdm(
                    range(1, C.CANDIDATE_SET_SIZE + 1), leave=False, disable=False
                )
                for c in num_candidates:
                    if _plot_stats:
                        # draugr.terminal_plot_stats_shared_x(stats, printer=train_session.write)
                        train_session.set_description(
                            f"Steps: {_step_i:9.0f}"
                            # f' | Ent: {stats.entropies.calc_moving_average():.2f}'
                        )
                        num_candidates.set_description(
                            f"Candidate #{c} of {C.CANDIDATE_SET_SIZE} | "
                            f"FP: {reached_dead_end} | "
                            f"L: {random_motion_length} | "
                            f"S_i: {len(S_initial)}"
                        )

                    seed = sample(S_prev)
                    S_candidate.extend(
                        environment.generate_trajectory_from_state(
                            seed, random_motion_length, random_process=_random_process
                        )
                    )

                    candidate = sample(S_candidate)

                    est, _episode_i, _step_i = estimate_initial_state_expected_return(
                        candidate,
                        environment,
                        agent,
                        C,
                        save_snapshot=save_snapshot,
                        # statistics=stats,
                        train=True,
                    )

                    if C.LOW <= est <= C.HIGH:
                        S_initial.append(candidate)
                        random_motion_length = C.RANDOM_MOTION_HORIZON
                        reached_dead_end = False
                    elif _keep_seed_if_not_replaced:
                        S_initial.append(seed)

                display_actor_configurations(environment, S_candidate)

                if reached_dead_end:
                    print("Reached dead end")
                    print("Generating initial state from goal configuration")
                    S_initial = environment.generate_trajectory_from_configuration(
                        initial_configuration,
                        random_motion_length,
                        random_process=_random_process,
                    )
                    random_motion_length += 1

                S_prev = S_initial

            time_elapsed = time.time() - training_start_timestamp
            message = (
                f"Time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )
            print(f'\n{"-" * 9} {message} {"-" * 9}\n')

            agent.save(C)
            save_snapshot()


if __name__ == "__main__":
    import neodroidagent.configs.curriculum.curriculum_config as C

    from neodroidagent.configs import parse_arguments

    args = parse_arguments("PG Agent", C)

    for key, arg in args.__dict__.items():
        setattr(C, key, arg)

    sprint(f"\nUsing config: {C}\n", highlight=True, color="yellow")
    if not args.skip_confirmation:
        for key, arg in get_upper_case_vars_or_protected_of(C).items():
            print(f"{key} = {arg}")
        input("\nPress Enter to begin... ")

    _agent = PGAgent(C)
    # _agent = DDPGAgent(C)
    # _agent = DQNAgent(C)

    try:
        main(C, _agent)
    except KeyboardInterrupt:
        print("Stopping")

    torch.cuda.empty_cache()
