import gzip
import pickle

import ecole
import numpy as np
import pyscipopt
import time
from pathlib import Path

import torch
from mpipool import MPIPool

import model
import utils
from bpp import BinpackingGenerator
from environments import HistoryBranchingEnvironment, SimpleBranching, SimpleConfiguring
from model import load_policy
from observer import UnrealisticObserver
from utils import get_ecole_model_from_file


class Solver:
    def __init__(self):
        self.model: pyscipopt.Model = None

    def solve(self, path: str):
        raise NotImplementedError

    def __getitem__(self, item):
        match item:
            case "time":
                return self.model.getSolvingTime()
            case "nnodes":
                return self.model.getNNodes()
            case "obj":
                return self.model.getObjVal()
            case "gap":
                return self.model.getGap()
            case _:
                raise KeyError


class EcoleSolver(Solver):

    def __init__(self, env: ecole.environment, before_start_callbacks=[], before_action_callbacks=[]):
        super().__init__()
        self.env: ecole.environment = env
        self.before_start_callbacks = before_start_callbacks
        self.before_action_callbacks = before_action_callbacks

    def set_before_action_callbacks(self, c):
        self.before_action_callbacks = c

    def set_before_start_callbacks(self, c):
        self.before_start_callbacks = c

    def solve(self, path: str):
        m = pyscipopt.Model()
        m.setParam("display/verblevel", 0)
        m.readProblem(path)
        instance: ecole.scip.Model = ecole.scip.Model.from_pyscipopt(m)
        self.before_start(path)
        self.env.seed(0)

        t = time.time()
        observation, action_set, _, done, info = self.env.reset(instance)
        self.model = instance.as_pyscipopt()

        nb_nodes = info["nb_nodes"]
        while not done:
            action = self.get_action(action_set, observation)
            self.before_action(action, action_set, observation)
            observation, action_set, _, done, info = self.env.step(action)
            nb_nodes += info["nb_nodes"]

        t = time.time() - t

        self.model = self.env.model.as_pyscipopt()

        return nb_nodes, t

    def before_start(self, path):
        return [f(path) for f in self.before_start_callbacks]

    def before_action(self, action, action_set, observation):
        return [f(action, action_set, observation) for f in self.before_action_callbacks]

    def get_action(self, action_set, observation) -> int:
        ub_scores, _ = observation[0]
        action_index = ub_scores[action_set].argmax()
        action = action_set[action_index]
        return action


class UBSolver(EcoleSolver):
    def __init__(self, n_repetitions=20, scip_params_env={}, scip_params={}, pool: MPIPool = None,
                 expert_probability=1.0,
                 additional_observers=[]):
        self.observer = UnrealisticObserver(
            n_repetitions,
            scip_params=scip_params,
            pool=pool,
            expert_probability=expert_probability,
        )

        env = HistoryBranchingEnvironment(
            observation_function=(
                self.observer,
                *additional_observers
            ),
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "time": ecole.reward.SolvingTime(),
            },
            scip_params=scip_params_env,
        )

        super().__init__(env)

    def before_action(self, action, action_set, observation):
        super().before_action(action, action_set, observation)
        self.observer.step(action)

    def before_start(self, path):
        super().before_start(path)
        self.observer.reset(path, 0)

    def __str__(self):
        return "UB"


class GnnSolver(EcoleSolver):
    def __init__(self, policy_path, scip_params={}):
        env = HistoryBranchingEnvironment(
            observation_function=ecole.observation.NodeBipartite(),
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "time": ecole.reward.SolvingTime(),
            },
            scip_params=scip_params,
        )
        self.policy = load_policy(policy_path)
        self.policy_path = policy_path

        super().__init__(env)

    def get_action(self, action_set, observation) -> int:
        device = model.get_device()
        observation = (
            torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
            torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
            torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device),
            torch.from_numpy(observation.variable_features.astype(np.float32)).to(device),
        )

        logits = self.policy(*observation)
        action = action_set[logits[action_set.astype(np.int64)].argmax()]
        return action

    def __str__(self):
        return "GNN({})".format(self.policy_path)


class ClassicSolver(Solver):
    def __init__(self, branching_policy, scip_params={}):
        super().__init__()
        self.model = pyscipopt.Model()
        self.branching_policy = branching_policy
        utils.configure(self.model, branching_policy=branching_policy)

        self.model.setParams(scip_params)

    def solve(self, path: str):
        self.model.readProblem(path)
        self.model.optimize()

    def __str__(self):
        return self.branching_policy
class DatasetGenerator:
    def __init__(self, instances, solver: EcoleSolver, sample_counter=0, episode_counter=0):
        self.solver = solver
        self.instances = instances
        self.sample_counter = sample_counter
        self.episode_counter = episode_counter

        for _ in range(self.episode_counter):
            next(self.instances)

    def generate(self, folder_name, max_samples):
        Path(folder_name).mkdir(parents=True, exist_ok=True)

        def before_action(action, action_set, observation):
            (scores, scores_are_expert), node_observation, *_ = observation

            if scores_are_expert and self.sample_counter < max_samples:
                self.sample_counter += 1
                data = [node_observation, action, action_set, scores]
                filename = f"{folder_name}/sample_{self.sample_counter}.pkl"

                with gzip.open(filename, "wb") as f:
                    pickle.dump(data, f)

        self.solver.set_before_action_callbacks([before_action])

        while self.sample_counter < max_samples:
            try:
                instance = next(self.instances)
                self.episode_counter += 1
                m = instance.as_pyscipopt()
                m.setParam("display/verblevel", 0)
                m.writeProblem("prob.lp")
                self.solver.solve("prob.lp")

                print(f"Episode {self.episode_counter}, {self.sample_counter} samples collected so far")
            except:
                print(f"Crash on episode {self.episode_counter}")


if __name__ == "__main__":
    n_objects = 20
    bin_capacity = 100
    weight_bounds = (20, 22)
    instances = BinpackingGenerator(n_objects, bin_capacity, weight_bounds)
    instances.seed(0)

    solver = UBSolver(n_repetitions=20, scip_params={"limits/nodes": 2000}, expert_probability=0.1,
                      additional_observers=[ecole.observation.NodeBipartite()])

    # generator = DatasetGenerator(instances, solver)
    # generator.generate("test", 100)

    # solver = UBSolver(20, {}, None)
    solver = ClassicSolver("vanillafullstrong")
    instance = next(instances)
    m = instance.as_pyscipopt()
    m.writeProblem("prob.lp")
    solver.solve("prob.lp")

    print(solver["time"], solver["gap"], solver["obj"], solver["nnodes"])
