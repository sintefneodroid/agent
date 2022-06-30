import argparse
import os
from itertools import count

import gym
import numpy
import torch
from torch import multiprocessing, nn, optim
from torch.distributed import rpc
from torch.distributed.rpc import RRef, remote, rpc_async, rpc_sync
from torch.distributions import Categorical
from torch.nn import functional

TOTAL_EPISODE_STEP = 5000
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"
ENV = "CartPole-v1"

parser = argparse.ArgumentParser(description="PyTorch RPC RL example")
parser.add_argument(
    "--world-size",
    type=int,
    default=2,
    metavar="W",
    help="world size for RPC, rank 0 is the agent, others are observers",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    metavar="G",
    help="discount factor (default: 0.99)",
)
parser.add_argument(
    "--seed", type=int, default=543, metavar="N", help="random seed (default: 543)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="interval between training status logs (default: 10)",
)
config = parser.parse_args()

torch.manual_seed(config.seed)


def _call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    return rpc_sync(
        rref.owner(), _call_method, args=(method, rref, *args), kwargs=kwargs
    )


class Policy(nn.Module):
    r"""
    Borrowing the ``Policy`` class from the Reinforcement Learning example.
    Copying the code to make these two examples independent.
    See https://github.com/pytorch/examples/tree/master/reinforcement_learning
    """

    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        return functional.log_softmax(
            self.affine2(functional.relu(self.dropout(self.affine1(x)))), dim=1
        )  # Check dim maybe it should be -1


class Observer:
    r"""
    An observer has exclusive access to its own environment. Each observer
    captures the state from its environment, and send the state to the agent to
    select an action. Then, the observer applies the action to its environment
    and reports the reward to the agent.

    It is true that CartPole-v1 is a relatively inexpensive environment, and it
    might be an overkill to use RPC to connect observers and trainers in this
    specific use case. However, the main goal of this tutorial to how to build
    an application using the RPC API. Developers can extend the similar idea to
    other applications with much more expensive environment.
    """

    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.env = gym.make(ENV)
        self.env.seed(config.seed)

    def run_episode(self, agent_rref, n_steps):
        r"""
        Run one episode of n_steps.

        Arguments:
            agent_rref (RRef): an RRef referencing the agent object.
            n_steps (int): number of steps in this episode
        """
        state, ep_reward = self.env.reset(), 0
        for step in range(n_steps):
            # send the state to the agent to get an action
            action = _remote_method(Agent.select_action, agent_rref, self.id, state)

            # apply the action to the environment, and get the reward
            state, reward, done, _ = self.env.step(action)

            # report the reward to the agent for training purpose
            _remote_method(Agent.report_reward, agent_rref, self.id, reward)

            if done:
                break


class Agent:
    def __init__(self, world_size):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.saved_log_probs = {}
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = numpy.finfo(numpy.float32).eps.item()
        self.running_reward = 0
        self.reward_threshold = gym.make(ENV).spec.reward_threshold
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer))
            self.rewards[ob_info.id] = []
            self.saved_log_probs[ob_info.id] = []

    def select_action(self, ob_id, state):
        r"""
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/master/reinforcement_learning
        The main difference is that instead of keeping all probs in one list,
        the agent keeps probs in a dictionary, one key per observer.

        NB: no need to enforce thread-safety here as GIL will serialize
        executions.
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        dist = Categorical(logits=self.policy(state))
        action = dist.sample()
        self.saved_log_probs[ob_id].append(dist.log_prob(action))
        return action.item()

    def report_reward(self, ob_id, reward):
        r"""
        Observers call this function to report rewards.
        """
        self.rewards[ob_id].append(reward)

    def run_episode(self, n_steps=0):
        r"""
        Run one episode. The agent will tell each oberser to run n_steps.
        """
        futs = []
        for (
            ob_rref
        ) in self.ob_rrefs:  # make async RPC to kick off an episode on all observers

            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    _call_method,
                    args=(Observer.run_episode, ob_rref, self.agent_rref, n_steps),
                )
            )

        for fut in futs:  # wait until all obervers have finished this episode
            fut.wait()

    def finish_episode(self) -> torch.Tensor:
        r"""
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/master/reinforcement_learning
        The main difference is that it joins all probs and rewards from
        different observers into one list, and uses the minimum observer rewards
        as the reward of the current episode.
        """

        # joins probs and rewards from different observers into lists
        R, probs, rewards = 0, [], []
        for ob_id in self.rewards:
            probs.extend(self.saved_log_probs[ob_id])
            rewards.extend(self.rewards[ob_id])

        min_reward = min(
            [sum(self.rewards[ob_id]) for ob_id in self.rewards]
        )  # use the minimum observer reward to calculate the running reward
        self.running_reward = 0.05 * min_reward + (1 - 0.05) * self.running_reward

        for ob_id in self.rewards:  # clear saved probs and rewards
            self.rewards[ob_id] = []
            self.saved_log_probs[ob_id] = []

        policy_loss, returns = [], []
        for r in rewards[::-1]:
            R = r + config.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for log_prob, R in zip(probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        return min_reward


def run_worker(rank, world_size) -> None:
    r"""
    This is the entry point for all processes. The rank 0 is the agent. All
    other ranks are observers.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if rank == 0:  # rank0 is the agent

        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)

        agent = Agent(world_size)
        for i_episode in count(1):
            n_steps = int(TOTAL_EPISODE_STEP / (config.world_size - 1))
            agent.run_episode(n_steps=n_steps)
            last_reward = agent.finish_episode()

            if i_episode % config.log_interval == 0:
                print(
                    f"Episode {i_episode}\tLast reward: {last_reward:.2f}\tAverage reward: {agent.running_reward:.2f}"
                )

            if agent.running_reward > agent.reward_threshold:
                print(f"Solved! Running reward is now {agent.running_reward}!")
                break
    else:  # other ranks are the observer
        rpc.init_rpc(
            OBSERVER_NAME.format(rank), rank=rank, world_size=world_size
        )  # observers passively waiting for instructions from agents

    rpc.shutdown()


if __name__ == "__main__":

    def main():
        multiprocessing.spawn(
            run_worker, args=(config.world_size,), nprocs=config.world_size, join=True
        )

    main()
