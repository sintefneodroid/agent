import random
from types import coroutine

import numpy as np
from neodroid.models import Configuration, Displayable, Reaction, ReactionParameters
from tqdm import tqdm


@coroutine
def grid_world_sample_entire_configuration_space(environment):
  if environment:
    actor_x_conf = environment.description.configurable('ActorTransformX_')
    # actor_y_conf = environment.description.configurable('ActorTransformY_')
    actor_z_conf = environment.description.configurable('ActorTransformZ_')
    x_space = actor_x_conf.configurable_space
    # y_space = actor_y_conf.configurable_space
    z_space = actor_z_conf.configurable_space
    for x in np.linspace(x_space.min_value, x_space.max_value, x_space.discrete_steps):
      for z in np.linspace(z_space.min_value, z_space.max_value, z_space.discrete_steps):
        # for y in np.linspace(y_space.min_value, y_space.max_value, y_space.discrete_steps):
        initial_configuration = [
          Configuration('ActorTransformX_', x),
          # Configuration('ActorTransformY_', y),
          Configuration('ActorTransformZ_', z),
          ]

        yield initial_configuration
  return


@coroutine
def grid_world_random_sample_uniformly_entire_configuration_space(environment):
  if environment:
    initial_configurations = []
    actor_x_conf = environment.description.configurable('ActorTransformX_')
    # actor_y_conf = environment.description.configurable('ActorTransformY_')
    actor_z_conf = environment.description.configurable('ActorTransformZ_')
    x_space = actor_x_conf.configurable_space
    # y_space = actor_y_conf.configurable_space
    z_space = actor_z_conf.configurable_space
    for x in np.linspace(x_space.min_value, x_space.max_value, x_space.discrete_steps):
      for z in np.linspace(z_space.min_value, z_space.max_value, z_space.discrete_steps):
        # for y in np.linspace(y_space.min_value, y_space.max_value, y_space.discrete_steps):
        initial_configuration = [
          Configuration('ActorTransformX_', x),
          # Configuration('ActorTransformY_', y),
          Configuration('ActorTransformZ_', z),
          ]
        initial_configurations.append(initial_configuration)

    while 1:
      yield random.sample(initial_configurations)
  return


def estimate_entire_state_space(env, agent, C, statistics, save_snapshot,
                                displayer_name='FullEvaluationPlotDisplayer'):
  actor_configurations = []
  success_estimates = []
  displayables = []
  for configuration in grid_world_sample_entire_configuration_space(env):
    configure_params = ReactionParameters(
        terminable=True,
        episode_count=False,
        reset=True,
        configure=True,
        )

    conf_reaction = Reaction(
        parameters=configure_params,
        configurations=configuration,
        displayables=displayables
        )

    displayables = [Displayable(displayer_name, (success_estimates, actor_configurations))]

    env.reset()
    state_ob, info = env.configure(conf_reaction)
    if not info.terminated:
      est, _, _ = estimate_value(info, env, agent, C, statistics, save_snapshot=save_snapshot, train=False)
      # TODO: Use a rollout of only policy sampled actions with no random sampling (effects like epsilon
      # exploration)

      vec3 = (configuration[0].configurable_value,
              0,  # configuration[1].configurable_value,
              configuration[1].configurable_value  # configuration[2].configurable_value
              )
      actor_configurations.append(vec3)
      success_estimates.append(est)

  displayables = [Displayable(displayer_name, (success_estimates, actor_configurations))]
  conf_reaction = Reaction(
      displayables=displayables
      )
  _ = env.configure(conf_reaction)


_episode_i = 0
_step_i = 0


def estimate_value(candidate, env, agent, C, statistics, save_snapshot, keep_stats=True, train=False):
  global _step_i, _episode_i

  rollout_signals = 0
  rollout_session = range(1, C.CANDIDATE_ROLLOUTS + 1)
  rollout_session = tqdm(rollout_session, leave=False)
  for j in rollout_session:
    rollout_session.set_description(
        f'Candidate rollout #{j} of {C.CANDIDATE_ROLLOUTS} | '
        f'Est: {rollout_signals / C.CANDIDATE_ROLLOUTS}'
        )
    state_ob, _ = env.configure(state=candidate)

    signals, steps, *stats = agent.rollout(state_ob, env, train=train)
    rollout_signals += signals

    if train:
      _step_i += steps
      _episode_i += 1

    if keep_stats:
      statistics.signals.append(signals)
      statistics.lengths.append(steps)
      statistics.entropies.append(stats[0])

    if _episode_i % C.SAVE_MODEL_INTERVAL == 0:
      pass
      # if keep_stats:
      #  save_snapshot()

  return rollout_signals / C.CANDIDATE_ROLLOUTS, _episode_i, _step_i


actor_configurations = []
success_estimates = []


def display_actor_configuration(env, candidate, frontier_displayer_name='FrontierPlotDisplayer'):
  actor_configuration = get_actor_configuration(env, candidate)
  vec3 = (actor_configuration[0], 0,
          actor_configuration[1])
  actor_configurations.append(vec3)
  est = 1
  success_estimates.append(est)
  frontier_displayable = [Displayable(frontier_displayer_name, (success_estimates, actor_configurations))]
  state_ob, info = env.display(frontier_displayable)


def get_initial_configuration(environment):
  state = environment.describe()
  if environment:
    goal_pos_x = environment.description.configurable('GoalTransformX_').configurable_value
    # goal_pos_y = environment.description.configurable('GoalTransformY_').configurable_value
    goal_pos_z = environment.description.configurable('GoalTransformZ_').configurable_value
    initial_configuration = [
      Configuration('ActorTransformX_', goal_pos_x),
      # Configuration('ActorTransformY_', goal_pos_y),
      Configuration('ActorTransformZ_', goal_pos_z),
      ]
    return initial_configuration


def get_actor_configuration(environment, candidate):
  state_ob, _ = environment.configure(state=candidate)
  # state = environment.describe()
  if environment:
    goal_pos_x = environment.description.configurable('ActorTransformX_').configurable_value
    # goal_pos_y = environment.description.configurable('ActorTransformY_').configurable_value
    goal_pos_z = environment.description.configurable('ActorTransformZ_').configurable_value
    return goal_pos_x, goal_pos_z
