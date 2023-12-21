import ecole.instance

from evaluation_tools import evaluate_solvers
from model import train
from solvers import *
import numpy as np

from uc import UcigGenerator


def main(pool: MPIPool = None):
    # Easy parameters
    #n_objects = 20
    #bin_capacity = 100
    #weight_bounds = (20, 22)

    #n_objects = 70
    #bin_capacity = 100
    #weight_bounds = (20, 80)

    #instances = BinpackingGenerator(n_objects, bin_capacity, weight_bounds)

    #instances = ecole.instance.SetCoverGenerator(n_rows=400, n_cols=800)

    instances = UcigGenerator("instances", start_at_instance=1000)
    #instances.seed(0)

    n_instances = 20

    scip_params = {
        "limits/nodes": 20,
    }

    #policy = train("sb_uc_5", 0.001, 20, "sb_uc_5.pkl")


    solvers = [
        ClassicSolver("vanillafullstrong", scip_params),
        ClassicSolver("relpscost", scip_params),
        UBSolver(20, scip_params_env=scip_params, scip_params={"limits/nodes": 100}, pool=pool),
        #ClassicSolver("random", scip_params),
        #GnnSolver("sb_uc_5.pkl", scip_params),
    ]


    metrics = ["nnodes", "time", "gap"]

    evaluate_solvers(solvers, instances, n_instances, metrics)





if __name__ == "__main__":
    t = time.time()
    utils.mpi_start(main)
    t = time.time() - t
    print(f"Runned in {round(t,2)} s")