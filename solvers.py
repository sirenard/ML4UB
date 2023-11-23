import gzip
import pickle

import ecole
import pyscipopt
import time
import numpy as np
from mpipool import MPIPool

from bpp import BinpackingGenerator
from environments import HistoryBranchingEnvironment
from observer import UnrealisticObserver
from utils import get_ecole_model_from_file


class Solver:
    def solve(self, path: str):
        raise NotImplementedError


class EcoleSolver(Solver):

    def __init__(self, env: ecole.environment, before_start_callbacks=[], before_action_callbacks=[]):
        self.env: ecole.environment = env
        self.before_start_callbacks = before_start_callbacks
        self.before_action_callbacks = before_action_callbacks

    def set_before_action_callbacks(self, c):
        self.before_action_callbacks = c

    def set_before_start_callbacks(self, c):
        self.before_start_callbacks = c

    def solve(self, path: str):
        instance = get_ecole_model_from_file(path)
        self.before_start(path)
        self.env.seed(0)

        t = time.time()
        observation, action_set, _, done, info = self.env.reset(instance)

        nb_nodes = info["nb_nodes"]
        while not done:
            action = self.get_action(action_set, observation)
            self.before_action(action, action_set, observation)
            observation, action_set, _, done, info = self.env.step(action)
            nb_nodes += info["nb_nodes"]

        t = time.time() - t

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
    def __init__(self, n_repetitions=20, scip_params={}, pool: MPIPool = None, expert_probability=1.0,
                 additionnal_observers=[]):
        self.observer = UnrealisticObserver(
            n_repetitions,
            scip_params=scip_params,
            pool=pool,
            expert_probability=expert_probability,
        )

        env = HistoryBranchingEnvironment(
            observation_function=(
                self.observer,
                *additionnal_observers
            ),
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "time": ecole.reward.SolvingTime(),
            },
            scip_params=scip_params,
        )

        super().__init__(env)

    def before_action(self, action, action_set, observation):
        super().before_action(action, action_set, observation)
        self.env.step(action)

    def before_start(self, path):
        super().before_start(path)
        self.observer.reset(path, 0)


class DatasetGenerator:
    def __init__(self, instances, solver: EcoleSolver):
        self.solver = solver
        self.instances = instances

    def generate(self, folder_name, max_samples):
        sample_counter = 0
        episode_counter = 0

        def before_action(action, action_set, observation):
            global sample_counter
            (scores, scores_are_expert), node_observation, *_ = observation

            if scores_are_expert and sample_counter < max_samples:
                sample_counter += 1
                data = [node_observation, action, action_set, scores]
                filename = f"{folder_name}/sample_{sample_counter}.pkl"

                with gzip.open(filename, "wb") as f:
                    pickle.dump(data, f)

        self.solver.set_before_action_callbacks([before_action])

        while sample_counter < max_samples:
            instance = next(self.instances)
            m = instance.as_pyscipopt()
            m.setParam("display/verblevel", 0)
            m.writeProblem("prob.lp")
            self.solver.solve("prob.lp")

            print(f"Episode {episode_counter}, {sample_counter} samples collected so far")
            episode_counter += 1


if __name__ == "__main__":
    n_objects = 20
    bin_capacity = 100
    weight_bounds = (20, 22)
    instances = BinpackingGenerator(n_objects, bin_capacity, weight_bounds)
    instances.seed(0)

    #solver = UBSolver(n_repetitions=20, scip_params={"limits/nodes": 2000}, expert_probability=0.1,
    #                  additionnal_observers=[ecole.observation.NodeBipartite()])


    #generator = DatasetGenerator(instances, solver)
    #generator.generate("test", 100)

    solver = UBSolver(20, {"limits/nodes": 2000}, None)
    instance = next(instances)
    m = instance.as_pyscipopt()
    m.writeProblem("prob.lp")
    solver.solve("prob.lp")
