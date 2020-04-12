from time import time

import numpy


class Trainer(object):
    def __init__(self, agent, env):
        """
An object to facilitate agent training and evaluation.

Parameters
----------
agent : :class:`NumpyAgent` instance
The agent to train.
env : ``gym.wrappers`` or ``gym.envs`` instance
The environment to run the agent on.
"""
        self.env = env
        self.agent = agent
        self.rewards = {"total": [], "smooth_total": [], "n_steps": [], "duration": []}

    def _train_episode(self, max_steps, render_every=None):
        t0 = time()
        if "train_episode" in dir(self.agent):
            # online training updates over the course of the episode
            reward, n_steps = self.agent.train_episode(max_steps)
        else:
            # offline training updates upon completion of the episode
            reward, n_steps = self.agent.run_episode(max_steps)
            self.agent.update()
        duration = time() - t0
        return reward, duration, n_steps

    def train(
        self,
        n_episodes,
        max_steps,
        seed=None,
        plot=True,
        verbose=True,
        render_every=None,
        smooth_factor=0.05,
    ):
        """
Train an agent on an OpenAI gym environment, logging training
statistics along the way.

Parameters
----------
n_episodes : int
The number of episodes to train the agent across.
max_steps : int
The maximum number of steps the agent can take on each episode.
seed : int or None
A seed for the random number generator. Default is None.
plot : bool
Whether to generate a plot of the cumulative reward as a function
of training episode. Default is True.
verbose : bool
Whether to print intermediate run statistics to stdout during
training. Default is True.
smooth_factor : float in [0, 1]
The amount to smooth the cumulative reward across episodes. Larger
values correspond to less smoothing.
"""
        if seed:
            numpy.random.seed(seed)
            self.env.seed(seed=seed)

        t0 = time()
        render_every = numpy.inf if render_every is None else render_every
        sf = smooth_factor

        for ep in range(n_episodes):
            tot_rwd, duration, n_steps = self._train_episode(max_steps)
            smooth_tot = tot_rwd if ep == 0 else (1 - sf) * smooth_tot + sf * tot_rwd

            if verbose:
                print(
                    f"[Ep. {ep + 1:2}] {n_steps:<6.2f} Steps | Total Reward: {tot_rwd:<7.2f} | Smoothed "
                    f"Total: {smooth_tot:<7.2f} | Duration: {duration:<6.2f}s"
                )

            if (ep + 1) % render_every == 0:
                total, n_steps = self.agent.greedy_policy(max_steps)
                print(
                    f"\tGreedy policy total reward: {total:.2f}, n_steps: {n_steps:.2f}"
                )

            self.rewards["total"].append(tot_rwd)
            self.rewards["n_steps"].append(n_steps)
            self.rewards["duration"].append(duration)
            self.rewards["smooth_total"].append(smooth_tot)

        train_time = (time() - t0) / 60
        print(
            f"Training took {train_time:.2f} mins [{numpy.mean(self.rewards['duration']):.2f}s/episode]"
        )

        rwd_greedy, n_steps = self.agent.greedy_policy(max_steps, render=False)
        print(f"Final greedy reward: {rwd_greedy:.2f} | n_steps: {n_steps:.2f}")

        if plot:
            self.plot_rewards(rwd_greedy)

    def plot_rewards(self, rwd_greedy):
        """
Plot the cumulative reward per episode as a function of episode number.

Notes
-----
Saves plot to the file ``./img/<agent>-<env>.png``

Parameters
----------
rwd_greedy : float
The cumulative reward earned with a final execution of a greedy
target policy.
"""
        try:
            from matplotlib import pyplot
            import seaborn as sns

            # https://seaborn.pydata.org/generated/seaborn.set_context.html
            # https://seaborn.pydata.org/generated/seaborn.set_style.html
            sns.set_style("white")
            sns.set_context("notebook", font_scale=1)
        except:
            fstr = "Error importing `matplotlib` and `seaborn` -- plotting functionality is disabled"
            raise ImportError(fstr)

        R = self.rewards
        fig, ax = pyplot.subplots()
        x = numpy.arange(len(R["total"]))
        y = R["smooth_total"]
        y_raw = R["total"]

        ax.plot(x, y, label="smoothed")
        ax.plot(x, y_raw, alpha=0.5, label="raw")
        ax.axhline(y=rwd_greedy, xmin=min(x), xmax=max(x), ls=":", label="final greedy")
        ax.legend()
        sns.despine()

        env = self.agent.env_info["id"]
        agent = self.agent.hyperparameters["agent"]

        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative reward")
        ax.set_title(f"{agent} on '{env}'")
        pyplot.savefig(f"img/{agent}-{env}.png")
        pyplot.close("all")
