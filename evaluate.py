import pickle

from evaluation_tools import evaluate_solvers
from mcts import MctsObserver
from model import train
from observer import LiftedObserver, StrongBranching, GnnObserver
from solvers import *

from uc import UcigGenerator


def main(pool = None):
    instances = UcigGenerator("instances", start_at_instance=5000)

    n_instances = 30

    scip_params = {
        "limits/nodes":20000,
    }

    scip_params_ub = {
        "limits/nodes": 5000,
    }

    #policy = train("ub_0_0.5", 0.001, 50, "ub_0_0.5.pkl")


    solvers = [
        ClassicSolver("vanillafullstrong", scip_params),
        ClassicSolver("relpscost", scip_params),
	    EcoleSolver(GnnObserver("sb_0.5.pkl"), scip_params=scip_params),
	    #GnnSolver("ub_0_0.5.pkl", scip_params),
        #LiftedSolver(scip_params, GnnObserver("sb_0.5.pkl", try_use_gpu=False), scip_params_ub, pool),
        #LiftedSolver(scip_params, StrongBranching(), scip_params_ub, pool),
        # MixEcoleSolver(
        #     [
        #         (LiftedSolver(scip_params, GnnObserver("sb_0.5.pkl", try_use_gpu=False), scip_params_ub, pool),
        #          lambda m: m.getNNodes()<=1),
        #         (GnnSolver("sb_0.5.pkl", scip_params),lambda m: True)
        #      ],
        #     scip_params
        # )
        #GnnSolver("ub_1_0.5.pkl", scip_params),
    ]


    metrics = ["nnodes", "time", "gap"]

    data = evaluate_solvers(solvers, instances, n_instances, metrics, catch_errors=False)
    pickle.dump(data, open("data", "wb"))

    data.performance_profile(metric="time")
    data.performance_profile(metric="nnodes")


if __name__ == "__main__":
    t = time.time()
    utils.mpi_start(main)
    t = time.time() - t
    print(f"Ran in {round(t,2)} s")