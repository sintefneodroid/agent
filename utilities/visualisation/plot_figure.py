import matplotlib.pyplot as plt
import numpy as np


def plot_figure(episodes, eval_rewards, env_id):
  episodes = np.array(episodes)
  eval_rewards = np.array(eval_rewards)
  np.savetxt("./output/%s_ppo_episodes.txt" % env_id, episodes)
  np.savetxt("./output/%s_ppo_eval_rewards.txt" % env_id, eval_rewards)

  plt.figure()
  plt.plot(episodes, eval_rewards)
  plt.title("%s" % env_id)
  plt.xlabel("Episode")
  plt.ylabel("Average Reward")
  plt.legend(["PPO"])
  plt.savefig("./output/%s_ppo.png" % env_id)
