import time

import ecole
import pyscipopt
from pyscipopt.scip import PY_SCIP_PARAMSETTING

import utils
from bpp import binpackingGenerator
from environments import HistoryBranchingEnvironment, SimpleConfiguring
from observer import UnrealisticObserver

n_instances = 5
n_repetitions = 100
n_cpus = 4

if __name__ == "__main__":
    n_objects = 15
    bin_capacity = 35
    weight_bounds = (5, 20)
    instances = binpackingGenerator(0, n_objects, bin_capacity, weight_bounds)

    observer = UnrealisticObserver(n_repetitions, n_cpus)
    env = HistoryBranchingEnvironment(
        observation_function=observer,
        information_function={
            "nb_nodes": ecole.reward.NNodes(),
            "time": ecole.reward.SolvingTime(),
        }
    )

    default_env = SimpleConfiguring(
        observation_function=None,
        information_function={
            "nb_nodes": ecole.reward.NNodes(),
            "time": ecole.reward.SolvingTime(),
        },
        scip_params={
            "branching/vanillafullstrong/priority": 99999999,
        },
    )

    for i, instance in zip(range(n_instances), instances):
        print(f"instance {i+1}/{n_instances}")
        instance.as_pyscipopt().writeProblem("prob.lp")

        observer.reset("prob.lp", 0)
        default_env.seed(0)
        env.seed(0)

        nb_nodes, t = 0, 0
        t = time.time()
        observation, action_set, _, done, info = env.reset(instance)
        nb_nodes += info["nb_nodes"]

        while not done:
            action_index = observation.argmin()
            action = action_set[action_index]

            observer.step(action)
            observation, action_set, _, done, info = env.step(action)

            nb_nodes += info["nb_nodes"]

        print(f"UB: {nb_nodes}")


        # Solve with ub
        model = instance.as_pyscipopt()
        utils.configure(model)
        nnodes, obj = utils.solve(model)

        print(f"SB: {nnodes}")
