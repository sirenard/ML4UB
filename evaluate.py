import pickle

import ecole.instance

import linear_model
import mcts
from evaluation_tools import evaluate_solvers
from model import train
import ranking_model
from observer import LiftedObserver, StrongBranching, GnnObserver, SVMObserver, LinearObserver
from solvers import *

from uc import UcigGenerator

ecole.observation.NodeBipartite


def main(pool = None):
    instances = UcigGenerator("instances", start_at_instance=5000, days=1)
    # instances = UcigGenerator("uc_hetrogeneous_10_1")

    # instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000)
    # instances = ecole.instance.IndependentSetGenerator()
    # instances = ecole.instance.CombinatorialAuctionGenerator()
    # instances.seed(12)

    n_instances = 10

    scip_params = {
        # "limits/nodes":20000,
        "limits/time": 300,
    }

    scip_params_ub = {
        "limits/nodes": 5000,
    }

    # policy = train("sc_0_speedup_0.6", 0.001, 15, "sc_0_speedup_0.6.pkl")



    solvers = [
        ClassicSolver("vanillafullstrong", scip_params),
        # ClassicSolver("relpscost", scip_params),
        # EcoleSolver(GnnObserver("sc_sb_0.05.pkl", try_use_gpu=True), scip_params=scip_params),
        # EcoleSolver(GnnObserver("sc_0_speedup_0.6.pkl", try_use_gpu=True), scip_params=scip_params),
        # EcoleSolver(GnnObserver("sb_0.5.pkl", try_use_gpu=True), scip_params=scip_params),
        # EcoleSolver(GnnObserver("ub_2_0.6_8.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("ub_1_speedup_0.6.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("ub_0_speedup_0.6.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("ub_1_speedup_0.6.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("ub_2_speedup_0.6.pkl", try_use_gpu=True), scip_params),
        EcoleSolver(mcts.MctsObserver(simulation_count=100, limits_rollout=6), scip_params),

        # EcoleSolver(
        #      LiftedObserver(StrongBranching(), scip_params_ub, pool),
        # ),
        # EcoleSolver(
        #     LiftedObserver(GnnObserver("sc_sb_0.05.pkl", try_use_gpu=False), scip_params_ub, pool),
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