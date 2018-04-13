import cv2
import torch
import torch as th
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from agents.ac_agent import ACAgent

cv2.setNumThreads(0)


import utilities as U


class PPOAgent(ACAgent):
  '''
  An agent learned with PPO using Advantage Actor-Critic framework
  - Actor takes state as input
  - Critic takes both state and action as input
  - agent interact with environment to collect experience
  - agent training with experience to update policy
  - adam seems better than rmsprop for ppo
  '''

  def __init__(self, state_dim, action_dim, config):

    super().__init__()
    self._rollout_i = 0
    self._step_i = 0
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.steps = config.STEPS

    self.gamma = config.GAMMA
    self.glp = config.GAE_LAMBDA_PARAMETER
    self.horizon_penalty = config.DONE_PENALTY

    self.experience_buffer = U.ExpandableBuffer()
    self.critic_loss = config.CRITIC_LOSS
    self.actor_critic_lr = config.ACTOR_LR
    self.critic_lr = config.CRITIC_LR
    self.entropy_reg_coef = config.ENTROPY_REG_COEF
    self.value_reg_coef = config.VALUE_REG_COEF
    self.batch_size = config.BATCH_SIZE
    self.episodes_before_train = config.EPISODES_BEFORE_TRAIN
    self.target_tau = config.TARGET_TAU
    self.max_grad_norm = config.MAX_GRADIENT_NORM

    # params for epsilon greedy
    self.epsilon_start = config.EPSILON_START
    self.epsilon_end = config.EPSILON_END
    self.epsilon_decay = config.EPSILON_DECAY

    self.use_cuda = config.use_cuda and th.cuda.is_available()

    self.update_target_interval = config.TARGET_UPDATE_STEPS
    self.clip = config.CLIP

    self.actor_critic = U.ActorCriticNetwork(config)
    self.actor_critic_target = U.ActorCriticNetwork(config)
    self.actor_critic_target.load_state_dict(self.actor_critic.state_dict())

    self.optimiser = config.OPTIMISER(self.actor_critic.parameters(), lr=self.actor_critic_lr)

    if self.use_cuda:
      self.actor_critic.cuda()
      self.actor_critic_target.cuda()

  def maybe_take_n_steps(self, initial_state, environment, n=100):
    state = initial_state
    accum_signal = 0

    transitions = []
    terminated = False

    T = tqdm(range(1, n + 1), f'Step #{self._step_i}', leave=False)
    for t in T:
      self._step_i += 1
      action, valuations, action_prob, *_ = self.continues_sample_model(state)

      next_state, signal, terminated, _ = environment.step(action)

      successor_state = None
      if not terminated:  # If environment terminated then there is no successor state
        successor_state = next_state

      transitions.append(
          U.ValuedTransition(state, action, action_prob, valuations, signal, successor_state,
                             not terminated))

      state = next_state

      accum_signal += signal

      if self._step_i % self.update_target_interval == 0:
        self.actor_critic_target.load_state_dict(self.actor_critic.state_dict())

      if terminated:
        break

    return transitions, accum_signal, terminated, state

  def trace_back_steps(self, transitions):
    n_step_summary = U.ValuedTransition(*zip(*transitions))
    signals = n_step_summary.signal
    value_estimates = n_step_summary.value_estimate

    advantages, discounted_returns = U.gae(signals,
                                           value_estimates,
                                           n_step_summary.non_terminal,
                                           self.gamma,
                                           glp=self.glp)  # compute GAE(lambda) advantages and discounted
    # returns

    i = 0
    advantage_memories = []
    for step in zip(*n_step_summary):
      step = U.ValuedTransition(*step)
      advantage_memories.append(
          U.AdvantageMemory(step.state,
                            step.action,
                            step.action_prob,
                            step.value_estimate,
                            advantages[i],
                            discounted_returns[i])
          )
      i += 1

    return advantage_memories

  def evaluate_model_error(self):
    batch = U.AdvantageMemory(*zip(*self.experience_buffer.memory))

    states_var = U.to_var(batch.state, use_cuda=self.use_cuda).view(-1, self.state_dim)
    action_var = U.to_var(batch.action, use_cuda=self.use_cuda, dtype='long').view(-1, 1)
    # action_probs_var = U.to_var(batch.action_prob, use_cuda=self.use_cuda).view(-1, self.action_dim)
    action_probs_var = U.to_var(batch.action_prob, use_cuda=self.use_cuda).view(-1, 1)
    values_var = U.to_var(batch.value_estimate, use_cuda=self.use_cuda).view(-1, 1)
    advantages_var = U.to_var(batch.advantage, use_cuda=self.use_cuda).view(-1, 1)
    returns_var = U.to_var(batch.discounted_return, use_cuda=self.use_cuda).view(-1, 1)

    # action_prob = action_probs_var.gather(1, action_var)
    action_prob = action_probs_var
    action_probs_t, _ = self.actor_critic_target(states_var)
    action_prob_t = action_probs_t.gather(1, action_var)
    ratio = action_prob / (action_prob_t + 1e-10)
    advantage = (advantages_var - advantages_var.mean()) / (advantages_var.std() + 1e-5)
    surrogate_1 = ratio * advantage
    surrogate_2 = torch.clamp(ratio, min=1. - self.clip, max=1. + self.clip) * advantage  # (L^CLIP)

    policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
    value_loss = (.5 * (values_var - returns_var) ** 2.).mean()
    entropy_loss = U.entropy(action_probs_var).mean()

    loss = policy_loss + value_loss * self.value_reg_coef + entropy_loss * self.entropy_reg_coef

    return loss

  def train(self):
    loss = self.evaluate_model_error()
    self.optimise_wrt(loss)

  def optimise_wrt(self, loss):
    self.optimiser.zero_grad()
    loss.backward()
    if self.max_grad_norm is not None:
      nn.utils.clip_grad_norm(self.actor_critic.parameters(), self.max_grad_norm)
    self.optimiser.step()

  def discrete_categorical_sample_model(self, state):
    state_var = U.to_var(state, use_cuda=self.use_cuda, unsqueeze=True)
    softmax_probs, value_estimate = self.actor_critic(state_var)
    m = Categorical(softmax_probs)
    action = m.sample()
    a = action.cpu().data.numpy()[0]
    return a, value_estimate, m.log_prob(action.data), m.probs

  def continues_sample_model(self, state):
    a_mean, a_log_std, value_estimate = self.actor_critic(state)

    # randomly sample from normal distribution, whose mean and variance come from policy network.
    # [b, a_dim]
    a = torch.normal(a_mean, torch.exp(a_log_std))

    # value, x, states = self(inputs, states, masks)
    # action = self.dist.sample(x, deterministic=deterministic)
    # action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)

    # return value, action, action_log_probs, states

    return a

  # choose an action based on state for execution
  def sample_action(self, state):
    action, *_ = self.discrete_categorical_sample_model(state)
    return action


def test_agent(env_id='CartPole-v0', seed=31):
  import gym
  import configs.ppo_config as c

  U.set_seed(seed)

  env = gym.make(env_id)
  env.seed(c.RANDOM_SEED)

  # env_fns = []
  # for rank in range(4):
  #  env_fns.append(lambda: make_env(env_id, rank, seed + rank))
  #  if False:
  #    env = RenderSubprocVecEnv(env_fns, 4)
  #  else:
  #    env = SubprocVecEnv(env_fns)

  # env = make_env(env_id, 0, seed)
  # env = VecFrameStack(env, 4)

  # test_env = make_env(env_id, 0, seed)
  # test_env = FrameStack(test_env, 4)
  print_interval = 10

  state_dim = env.observation_space.shape[0]
  if len(env.action_space.shape) > 1:
    action_dim = env.action_space.shape[0]
  else:
    action_dim = env.action_space.n

  c.ARCH_PARAMS['input_size'] = [state_dim]
  c.ARCH_PARAMS['output_size'] = [action_dim]

  ppo_agent = PPOAgent(state_dim, action_dim, c)

  initial_state = env.reset()
  cs = tqdm(range(1, c.ROLLOUTS + 1), f'Rollout {0}, {0}', leave=True)
  for rollout_i in cs:

    transitions, accum_signal, terminated, initial_state = ppo_agent.maybe_take_n_steps(initial_state, env)

    if terminated:
      initial_state = env.reset()

    if rollout_i >= c.EPISODES_BEFORE_TRAIN:
      advantage_memories = ppo_agent.trace_back_steps(transitions)
      for m in advantage_memories:
        ppo_agent.experience_buffer.remember(m)
      ppo_agent.train()
      ppo_agent.experience_buffer.forget()

    if rollout_i % print_interval == 0:
      cs.set_description(f'Rollout {rollout_i}, {accum_signal}')


if __name__ == '__main__':
  test_agent()
