"""Covariance Matrix Adaptation Evolution Strategy."""
from typing import Sequence

import numpy
from cma import CMAEvolutionStrategy
from draugr.torch_utilities import TensorBoardPytorchWriter
from draugr.writers import MockWriter, Writer
from garage.envs import GarageEnv
from garage.experiment import SnapshotConfig
from garage.sampler import LocalSampler, RaySampler
from garage.tf.policies import CategoricalMLPPolicy
from neodroidagent import PROJECT_APP_PATH
from neodroidagent.agents.numpy_agents.evolutionary.get_rid.local_tf_runner import (
    LocalTFRunner,
)
from neodroidagent.agents.numpy_agents.evolutionary.get_rid.meh_wat import (
    TrajectoryBatch,
)
from neodroidagent.agents.numpy_agents.model_free.baseline.linear_feature_gae_estimator import (
    LinearFeatureBaseline,
)
from tensorflow import Module
from warg import GDKC


class CovarianceMatrixAdaptationEvolutionStrategyAgent:
    """Covariance Matrix Adaptation Evolution Strategy.
    Note:
        The CMA-ES method can hardly learn a successful policy even for
        simple task. It is still maintained here only for consistency with
        original rllab paper.
    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy_arch (garage.numpy.policies.Policy): Action policy.
        baseline (garage.numpy.baselines.Baseline): Baseline for GAE
            (Generalized Advantage Estimation).
        num_candidate_policies (int): Number of policies sampled in one epoch.
        discount_factor (float): Environment reward discount.
        max_rollout_length (int): Maximum length of a single rollout.
        parameters_variance (float): Initial std for param distribution.
    """

    def __init__(
        self,
        env_spec,
        num_candidate_policies: int = 20,
        policy_arch: GDKC = GDKC(CategoricalMLPPolicy, hidden_sizes=(32, 32)),
        baseline_arch: GDKC = GDKC(
            LinearFeatureBaseline
        ),  # Baseline for GAE (Generalized Advantage Estimation).
        discount_factor: float = 0.99,
        max_rollout_length: int = 500,
        parameters_variance: float = 1.0,
    ):
        self.policy: Module = policy_arch(env_spec=env_spec)
        self.max_path_length = max_rollout_length  # TODO: REMOVE THIS..
        self.sampler_cls = RaySampler

        self._baseline = baseline_arch()
        self._max_rollout_length = max_rollout_length

        self._env_spec = env_spec
        self._discount = discount_factor
        self._parameters_variance = parameters_variance
        self._num_candidate_policies = num_candidate_policies

        self._evolution_strategy: CMAEvolutionStrategy = None
        self._shared_params = None

        self._all_returns = None

    def _resample_shared_parameters(self) -> None:
        """Return sample parameters.
        Returns:
            numpy.ndarray: A numpy array of parameter values.
        """
        self._shared_params = self._evolution_strategy.ask()

    def build(self):
        """ """
        pass  # TODO:

    def __build__(self, init_mean_parameters: Sequence):
        self._evolution_strategy = CMAEvolutionStrategy(
            init_mean_parameters,
            self._parameters_variance,  # Sigma is shared
            {"popsize": self._num_candidate_policies},
        )  # Population size
        self._resample_shared_parameters()
        self.policy.set_param_values(self._shared_params[0])

    def train(self, runner):
        """Initialize variables and start training.
        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.
        Returns:
            float: The average return in last epoch cycle.
        """
        self.__build__(self.policy.get_param_values())

        self._all_returns = []

        # start actual training
        last_return = None

        for _ in runner.step_epochs():
            for _ in range(self._num_candidate_policies):
                runner.step_path = runner.obtain_samples(runner.step_itr)
                last_return = self.train_once(runner.step_itr, runner.step_path)
                runner.step_itr += 1

        return last_return

    def extract_signal(self):
        """ """
        pass

    def train_once(
        self,
        iteration_number: int,
        trajectories: Sequence,
        *,
        writer: Writer = MockWriter()
    ):
        """Perform one step of policy optimization given one batch of samples.
        Args:
            iteration_number (int): Iteration number.
            trajectories (list[dict]): A list of collected paths.
        Returns:
            float: The average return in last epoch cycle.
            :param iteration_number:
            :type iteration_number:
            :param trajectories:
            :type trajectories:
            :param writer:
            :type writer:
        """

        undiscounted_returns = []
        for trajectory in TrajectoryBatch.from_trajectory_list(
            self._env_spec, trajectories
        ).split():  # TODO: EEEEW
            undiscounted_returns.append(sum(trajectory.rewards))

        sample_returns = numpy.mean(undiscounted_returns)
        self._all_returns.append(sample_returns)

        epoch = iteration_number // self._num_candidate_policies
        i_sample = iteration_number - epoch * self._num_candidate_policies
        writer.scalar("Epoch", epoch)
        writer.scalar("# Sample", i_sample)

        if (
            iteration_number + 1
        ) % self._num_candidate_policies == 0:  # When looped all the way around update shared parameters, WARNING RACE CONDITIONS!
            sample_returns = max(self._all_returns)
            self.update()

        self.policy.set_param_values(
            self._shared_params[(i_sample + 1) % self._num_candidate_policies]
        )

        return sample_returns


def update(self) -> None:
    """ """
    self._evolution_strategy.tell(
        self._shared_params, -numpy.array(self._all_returns)
    )  # Report back results
    self.policy.set_param_values(
        self._evolution_strategy.best.get()[0]
    )  # TODO: DOES NOTHING, as is overwritten everywhere

    self._all_returns.clear()  # Clear for next epoch
    self._resample_shared_parameters()


if __name__ == "__main__":
    path = PROJECT_APP_PATH.user_data / "data" / "local" / "experiment"
    snapshot_config = SnapshotConfig(
        snapshot_dir=path, snapshot_mode="last", snapshot_gap=1
    )

    def stest_cma_es_cartpole():
        """Test CMAES with Cartpole-v1 environment."""
        with LocalTFRunner(snapshot_config) as runner:
            with TensorBoardPytorchWriter(PROJECT_APP_PATH.user_log / "CMA") as writer:
                env = GarageEnv(env_name="CartPole-v1")

                algo = CovarianceMatrixAdaptationEvolutionStrategyAgent(
                    env_spec=env.spec, max_rollout_length=100
                )
                algo.build()

                runner.setup(algo, env, sampler_cls=LocalSampler)
                runner.train(n_epochs=1, batch_size=1000)

                env.close()

    stest_cma_es_cartpole()
