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
    # instances = UcigGenerator("instances", start_at_instance=0, days=2)
    instances = UcigGenerator("uc_hetrogeneous_5_1", start_at_instance=0)

    # instances = ecole.instance.SetCoverGenerator(n_rows=400, n_cols=1000)
    # instances = ecole.instance.CapacitatedFacilityLocationGenerator()
    # instances = ecole.instance.CombinatorialAuctionGenerator()
    # instances.seed(0)

    instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05)
    instances.seed(0)

    n_instances = 10

    scip_params = {
        #"estimation/method": "c",
        # "estimation/completiontype": "m",
        # "limits/nodes":20000,
        # "limits/time": 300,
    }

    # policy = train("uc_hetrogeneous_5_1_ub_1_1_2", 0.005, 30, "uc_hetrogeneous_5_1_ub_1_1_2.pkl")
    # policy = train("cfl_sb_1", 0.005, 30, "cfl_sb_1.pkl")
    # policy = train("uc_hetrogeneous_5_1_ub_1_0.6", 0.001, 15, "uc_hetrogeneous_5_1_ub_1_0.6.pkl")



    solvers = [
        # ClassicSolver("vanillafullstrong", scip_params),
        ClassicSolver("relpscost", scip_params),
        # EcoleSolver(GnnObserver("uc_hetrogeneous_5_1_ub_0_1.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_hetrogeneous_5_1_sb_1.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_hetrogeneous_5_1_ub_0_1_2.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_hetrogeneous_5_1_ub_1_1_2.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_hetrogeneous_5_1_ub_0_1.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_hetrogeneous_5_1_ub_2_0.6.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("cfl_sb_1.pkl", try_use_gpu=True), scip_params),

        # EcoleSolver(
        #      LiftedObserver(StrongBranching(), scip_params, pool, node_limit=500,  score_method="speedup", local=True),
        # ),
        # EcoleSolver(
        #     LiftedObserver(StrongBranching(), scip_params, pool, node_limit=500, score_method="speedup", local=False),
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