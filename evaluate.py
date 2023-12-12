import ecole.instance

from evaluation_tools import evaluate_solvers
from solvers import *
import numpy as np

def main(pool: MPIPool = None):
    # Easy parameters
    #n_objects = 20
    #bin_capacity = 100
    #weight_bounds = (20, 22)

    #n_objects = 70
    #bin_capacity = 100
    #weight_bounds = (20, 80)

    #instances = BinpackingGenerator(n_objects, bin_capacity, weight_bounds)

    instances = ecole.instance.SetCoverGenerator(n_rows=400, n_cols=800)

    instances.seed(0)

    n_instances = 20

    scip_params = {
        "limits/nodes": 100000,
    }

    solvers = [
        #UBSolver(20, pool=pool),
        #ClassicSolver("vanillafullstrong", scip_params),
        #ClassicSolver("relpscost", scip_params),
        ClassicSolver("random", scip_params),
        #GnnSolver("agent_ub_bpp.pkl", scip_params),
    ]


    metrics = ["nnodes", "time", "gap"]

    evaluate_solvers(solvers, instances, n_instances, metrics)





if __name__ == "__main__":
    t = time.time()
    utils.mpi_start(main)
    t = time.time() - t
    print(f"Runned in {round(t,2)} s")