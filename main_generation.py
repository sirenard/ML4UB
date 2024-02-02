import ecole.instance

from datasetGenerator import DatasetGenerator
from mpi import MpiPool
from observer import LiftedObserver, StrongBranching
from solvers import *

from uc import UcigGenerator


def main(pool: MpiPool = None):
    instances = UcigGenerator("instances")

    expert_probability = 0.5

    scip_params = {
        "limits/nodes": 200,
    }

    scip_params_ub = {
        "limits/nodes": 500
    }


    generator = DatasetGenerator(
        instances,
        expert_observer=LiftedObserver(observer=StrongBranching(), scip_params=scip_params_ub, pool=pool),
        exploration_observer=ecole.observation.Pseudocosts(),
        expert_probability=expert_probability,
        scip_params=scip_params,
        )
    generator.generate(f"ub_0_{expert_probability}", max_instances=1000)



if __name__ == "__main__":
    t = time.time()
    utils.mpi_start(main)
    t = time.time() - t
    print(f"Runned in {round(t, 2)} s")
