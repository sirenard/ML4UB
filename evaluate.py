import pickle

import ecole.instance

import linear_model
from evaluation_tools import evaluate_solvers
from model import train
import ranking_model
from observer import LiftedObserver, StrongBranching, GnnObserver, SVMObserver, LinearObserver
from solvers import *

from uc import UcigGenerator


def main(pool = None):
    instances = UcigGenerator("instances", start_at_instance=5000, days=1)
    #instances = ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=90, n_facilities=90)

    n_instances = 30

    scip_params = {
        "limits/nodes":20000,
        #"limits/time": 30,
    }

    scip_params_ub = {
        "limits/nodes": 5000,
    }

    #policy = train("ub_2_0.6", 0.001, 15, "ub_2_0.6.pkl")
    #policy = train("ub_0_1_limit_20_alpha2_none", 0.001, 15, "ub_0_1_limit_20_alpha2_none.pkl")
    #policy = train("ub_0_1_limit_20_alpha2_0.1", 0.001, 15, "ub_0_1_limit_20_alpha2_0.1.pkl")
    #rank_model = ranking_model.train("test", "test.pkl")
    #model = linear_model.train("test", "test.pkl")


    solvers = [
        ClassicSolver("vanillafullstrong", scip_params),
        #ClassicSolver("relpscost", scip_params),
        #EcoleSolver(GnnObserver("sb_0.5.pkl", try_use_gpu=True), scip_params=scip_params),
        EcoleSolver(GnnObserver("ub_1_0.6.pkl", try_use_gpu=True), scip_params),
        EcoleSolver(GnnObserver("ub_2_0.6_8.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("ub_0_1_limit_20_alpha2_0.1.pkl", try_use_gpu=False), scip_params),

        #EcoleSolver(LinearObserver("test.pkl"), scip_params),
        #EcoleSolver(SVMObserver("test.pkl"), scip_params),

        # EcoleSolver(
        #      LiftedObserver(StrongBranching(), scip_params_ub, pool),
        # ),
        # EcoleSolver(
        #     LiftedObserver(GnnObserver("ub_1_0.6.pkl", try_use_gpu=False), scip_params_ub, pool),
        # ),
    ]


    metrics = ["nnodes", "time", "gap"]

    data = evaluate_solvers(solvers, instances, n_instances, metrics)
    pickle.dump(data, open("data", "wb"))

    data.performance_profile(metric="time")
    data.performance_profile(metric="nnodes")


if __name__ == "__main__":
    t = time.time()
    utils.mpi_start(main)
    t = time.time() - t
    print(f"Ran in {round(t,2)} s")