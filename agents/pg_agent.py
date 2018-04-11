import time
from itertools import count

import gym
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from utilities.visualisation.term_plot import term_plot

tqdm.monitor_interval = 0

import utilities as U

from torch.distributions import Categorical
from agents.policy_agent import PolicyAgent


class PGAgent(PolicyAgent):
  def __init__(self, config, gamma=0.99, lr=1e-4, entropy_reg=1e-4):

    super().__init__()
    self.C = config
    self._gamma = gamma
    self._learning_rate = lr
    self._trajectory = U.Trajectory()
    self._entropy_reg = entropy_reg
    self._signal_clipping = config.SIGNAL_CLIPPING

    self._optimiser_weight_decay = config.OPTIMISER_WEIGHT_DECAY
    self._evaluation_function = config.EVALUATION_FUNCTION()
    self._rollout_i = 0

    self._use_batched_updates = False  # self.C.BATCHED_UPDATES
    self._batch_size = 5  # self.C.BATCH_SIZE
    self._accum_error = U.to_var([0.0], use_cuda=self.C.USE_CUDA_IF_AVAILABLE)

  def build_model(self, env):
    if type(self.C.POLICY_ARCH_PARAMS['input_size']) == str:
      self.C.POLICY_ARCH_PARAMS['input_size'] = env.observation_space.shape
    print('observation dimensions: ', self.C.POLICY_ARCH_PARAMS['input_size'])

    if type(self.C.POLICY_ARCH_PARAMS['output_size']) == str:
      if hasattr(env.action_space, 'num_binary_actions'):
        self.C.POLICY_ARCH_PARAMS['output_size'] = [env.action_space.num_binary_actions]
      else:
        self.C.POLICY_ARCH_PARAMS['output_size'] = [env.action_space.n]
    print('action dimensions: ', self.C.POLICY_ARCH_PARAMS['output_size'])

    self._model = self.__build_model__(self.C)

  def __build_model__(self, C):

    model = C.POLICY_ARCH(**self.C.POLICY_ARCH_PARAMS)

    if self.C.USE_CUDA_IF_AVAILABLE:
      model = model.cuda()

    self.optimiser = C.OPTIMISER_TYPE(model.parameters(), lr=self._learning_rate,
                                      weight_decay=self._optimiser_weight_decay)

    return model

  def sample_action(self, state):
    state_var = U.to_var(state, use_cuda=self.C.USE_CUDA_IF_AVAILABLE, unsqueeze=True)
    probs = self._model(state_var)
    m = Categorical(probs)
    action = m.sample()
    a = action.cpu().data.numpy()[0]
    # a = np.argmax(a)
    return a, m.log_prob(action), U.log_entropy(probs)

  def sample_cont_action(self, state):
    state_var = U.to_var(state, use_cuda=self.C.USE_CUDA_IF_AVAILABLE, unsqueeze=True)
    mu, sigma_sq = self._model(state_var)  # requires MultiheadedMLP

    eps = torch.randn(mu.size())
    # calculate the probability
    action = (mu + sigma_sq.sqrt() * Variable(eps).cuda()).data
    prob = U.normal(action, mu, sigma_sq)
    entropy = -0.5 * ((sigma_sq + 2 * U.torch_pi.expand_as(sigma_sq)).log() + 1)

    log_prob = prob.log()
    return action, log_prob, entropy

  def evaluate_model(self):
    R = 0
    policy_loss = []
    signals = []
    for r in self._trajectory.signals[::-1]:
      R = r + self._gamma * R
      signals.insert(0, R)

    signals = U.to_tensor(signals, use_cuda=self.C.USE_CUDA_IF_AVAILABLE)

    stddev = signals.std()  # + np.finfo(np.float32).eps) for no zero division
    if signals.shape[0] > 1 and stddev > 0:
      signals = (signals - signals.mean()) / stddev
    else:
      return None

    for log_prob, signal, entropy in zip(self._trajectory.log_probs, signals, self._trajectory.entropies):
      policy_loss.append(-log_prob * signal - self._entropy_reg * entropy)

    loss = torch.cat(policy_loss).sum()
    return loss

  def rollout(self, initial_state, environment, render=False):
    self._rollout_i += 1

    episode_signal = 0
    episode_length = 0
    episode_entropy = 0

    state = initial_state

    T = count(1)
    T = tqdm(T, f'Rollout #{self._rollout_i}', leave=False)

    for t in T:
      action, action_log_probs, entropy, *_ = self.sample_action(state)

      state, signal, terminated, info = environment.step(action)

      if self._signal_clipping:
        signal = np.clip(signal, -1.0, 1.0)

      episode_signal += signal
      episode_entropy += entropy.data.cpu().numpy()
      self._trajectory.remember(signal, action_log_probs, entropy)

      if render:
        environment.render()

      if terminated:
        episode_length = t
        break

    error = self.evaluate_model()
    self._trajectory.forget()
    if error is not None:
      if self._use_batched_updates:
        self._accum_error += error
        if self._rollout_i % self._batch_size == 0:
          self.optimise_wrt(self._accum_error / self._batch_size)
          self._accum_error = U.to_var([0.0], use_cuda=self.C.USE_CUDA_IF_AVAILABLE)
      else:
        self.optimise_wrt(error)

    return episode_signal, episode_length, episode_entropy / episode_length

  def optimise_wrt(self, loss):
    self.optimiser.zero_grad()
    loss.backward()
    for params in self._model.parameters():
      params.grad.data.clamp_(-1, 1)
    self.optimiser.step()

  def save_model(self, C):
    U.save_model(self._model, C)

  def load_model(self, C, model_path):
    print('loading latest model: ' + model_path)
    self._model = C.POLICY_ARCH(**C.POLICY_ARCH_PARAMS)
    self._model.load_state_dict(torch.load(model_path))
    self._model = self._model.eval()  # _model.train(False)
    if C.USE_CUDA_IF_AVAILABLE:
      self._model = self._model.cuda()

  def infer(self, env, render=True):

    for episode_i in count(1):
      print('Episode {}'.format(episode_i))
      state = env.reset()

      for episode_frame_i in count(1):

        action, *_ = self.sample_action(state)
        state, reward, terminated, info = env.step(action)
        if render:
          env.render()

        if terminated:
          break

  def train(self, _environment, rollouts=1000, render=False, render_frequency=100, stat_frequency=100):

    training_start_timestamp = time.time()
    E = range(1, rollouts)
    E = tqdm(E, f'Episode: {1}', leave=True)

    running_length = 0
    running_lengths = []
    for episode_i in E:
      initial_state = _environment.reset()

      if episode_i % stat_frequency == 0:
        x_t = [i for i in range(1, episode_i + 1)]
        term_plot(x_t, running_lengths, 'Running Lengths', printer=E.write)
        E.set_description(f'Episode: {episode_i}, Running length: {running_length}')

      if render and episode_i % render_frequency == 0:
        signal, dur, *stats = self.rollout(initial_state, _environment, render=render)
      else:
        signal, dur, *stats = self.rollout(initial_state, _environment)

      running_length = running_length * 0.99 + dur * 0.01
      running_lengths.append(running_length)

    time_elapsed = time.time() - training_start_timestamp
    end_message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed %60:.0f}s'
    print('\n{} {} {}\n'.format('-' * 9, end_message, '-' * 9))

    return self._model, []


def test_pg_agent():
  import configs.pg_config as C

  env_name = 'CartPole-v0'
  # env_name = 'LunarLander-v2' # (coord_x, coord_y, vel_x, vel_y, angle, angular_vel, l_leg_on_ground,
  # r_leg_on_ground)

  env = gym.make(env_name)
  env.seed(C.RANDOM_SEED)

  torch.manual_seed(C.RANDOM_SEED)
  C.USE_CUDA_IF_AVAILABLE = False
  C.ARCH_PARAMS['input_size'] = [4]
  C.ARCH_PARAMS['output_size'] = [env.action_space.n]
  agent = PGAgent(C)
  agent.build_model(env)

  agent = PGAgent(C)
  agent.build_model(env)

  _trained_model, training_statistics, *_ = agent.train(env, 10000, render=False)
  U.save_model(_trained_model, C)

  env.close()


#
# def test_agent():
#   import configs.pg_config as C
#
#   env_name = 'CartPole-v0' # 'LunarLander-v2'
#   env = gym.make(env_name)
#   env.seed(C.RANDOM_SEED)
#
#   torch.manual_seed(C.RANDOM_SEED)
#   C.USE_CUDA_IF_AVAILABLE = False
#   C.ARCH_PARAMS['input_size'] = [4]
#   C.ARCH_PARAMS['output_size'] = [env.action_space.n]
#   agent = PGAgent(C)
#   agent.build_model(env)
#   render = False
#
#   running_length = 0
#   for i_episode in count(1):
#     state = env.reset()
#
#     episode_length=0
#     T=range(10000)
#     for t in T:
#       action, log_prob, entropy, *_ = agent.sample_action(state)
#       state, reward, done, _ = env.step(action)
#
#       if render:
#         env.render()
#
#       agent.trajectory.remember(reward, log_prob, entropy)
#       if done:
#         episode_length = t
#         break
#
#     running_length = running_length * 0.99 + episode_length * 0.01
#     error = agent.evaluate_model()
#     agent.trajectory.forget()
#     if error is not None:
#       if agent.use_batched_updates:
#         agent.accum_error += error
#         if agent._rollout_i % agent.batch_size == 0:
#           agent.optimise_wrt(agent.accum_error / agent.batch_size)
#           agent.accum_error = U.to_var([0.0], use_cuda=agent.C.USE_CUDA_IF_AVAILABLE)
#       else:
#         agent.optimise_wrt(error)
#
#     if i_episode % 10 == 0:
#       print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
#           i_episode, episode_length, running_length))
#
#     if running_length > env.spec.reward_threshold:
#       print("Solved! Running reward is now {} and "
#             "the last episode runs to {} time steps!".format(running_length,
#                                                              episode_length))
#       torch.save(agent._model.state_dict(), 'policy.model')
#       break


if __name__ == '__main__':
  test_pg_agent()