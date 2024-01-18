import ecole.instance

from datasetGenerator import DatasetGenerator
from observer import LiftedObserver
from solvers import *

from uc import UcigGenerator


def main(pool: MpiPool = None):
    instances = UcigGenerator("instances")

    expert_probability = 0.5

    scip_params = {
        "limits/nodes": 1000,
    }

    scip_params_ub = {
        "limits/nodes": 2000
    }


    generator = DatasetGenerator(
        instances,
        expert_observer=LiftedObserver(observer=GnnObserver("sb_0.5.pkl"), scip_params=scip_params_ub, pool=pool),
        exploration_observer=ecole.observation.StrongBranchingScores(),
        expert_probability=expert_probability,
        scip_params=scip_params,
        )
    generator.generate(f"ub_1_{expert_probability}", max_instances=1000)



if __name__ == "__main__":
    t = time.time()
    utils.mpi_start(main)
    t = time.time() - t
    print(f"Runned in {round(t, 2)} s")
