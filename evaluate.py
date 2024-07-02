import pickle

import ecole.instance

from evaluation_tools import evaluate_solvers
from observer import LiftedObserver, StrongBranching, GnnObserver, SVMObserver, LinearObserver, EntropyObserver, \
    ConditionalObservers, RandomObserver, NodeBipartiteExtended
from solvers import *

from uc import UcigGenerator



if __name__ == '__main__':
    instances = UcigGenerator("instances", start_at_instance=2500, days=3)
    # instances = UcigGenerator("uc_hetrogeneous_10_1", start_at_instance=5000)

    # instances = ecole.instance.SetCoverGenerator(n_rows=400, n_cols=1000)
    # instances = ecole.instance.CapacitatedFacilityLocationGenerator()
    # instances = ecole.instance.CombinatorialAuctionGenerator( n_items=200, n_bids=1000)
    # instances.seed(666)

    # instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05)
    # instances = ecole.instance.IndependentSetGenerator()
    # instances.seed(666)

    n_instances = 10

    scip_params = {
        "estimation/method": "c",
        "estimation/completiontype": "m",
        # "limits/nodes":1000,
        "limits/time": 300,
    }

    # policy = train("uc_5_1_ub_2_0.6_local", 0.005, 30, "uc_5_1_ub_1_0.6_local.pkl")
    # policy = train("ca_ub_0_1", 0.005, 20, "ca_ub_0_1.pkl")
    # policy = train("sc_500_1000_ub_0_1_local", 0.005, 30, "sc_500_1000_ub_0_1_local.pkl")



    solvers = [
        # ClassicSolver("vanillafullstrong", scip_params),
        # EcoleSolver(StrongBranching(), scip_params),
        ClassicSolver("relpscost", scip_params),
            # EcoleSolver(GnnObserver("uc_hetrogeneous_5_1_sb_1.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_hetrogeneous_5_1_ub_0_1_2.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_hetrogeneous_5_1_ub_1_1_2.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_hetrogeneous_5_1_ub_0_1.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("sc_500_1000_ub_0_1_local.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_hetrogeneous_5_1_sb_1.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_hetrogeneous_5_1_ub_1_1_local.pkl", try_use_gpu=True), scip_params),
        EcoleSolver(GnnObserver("uc_5_1_sb_0.6.pkl", try_use_gpu=True), scip_params),
        EcoleSolver(GnnObserver("uc_5_1_ub_2_1_local.pkl", try_use_gpu=True), scip_params),
        EcoleSolver(GnnObserver("uc_5_1_ub_1_0.6_local.pkl", try_use_gpu=True), scip_params),
        EcoleSolver(GnnObserver("uc_5_1_ub_2_0.6_local.pkl", try_use_gpu=True), scip_params),

        # EcoleSolver(GnnObserver("uc_hetrogeneous_5_1_sb_1_extended.pkl", try_use_gpu=True, feature_observer=NodeBipartiteExtended(), n_tree_features=1), scip_params),
        # EcoleSolver(EntropyObserver(n_solve=20)),
        # EcoleSolver(
        #     ConditionalObservers(
        #     [GnnObserver("uc_hetrogeneous_5_1_ub_2_1_local.pkl", try_use_gpu=True), GnnObserver("uc_hetrogeneous_5_1_sb_1.pkl", try_use_gpu=True)],
        #     [lambda m: m.getNNodes()<=200, lambda m: True]
        #     )
        # ),

        # EcoleSolver(
        #     LiftedObserver(StrongBranching(), scip_params, pool, node_limit=100, score_method="speedup", local=True),
        #     scip_params
        # ),
        # EcoleSolver(
        #     LiftedObserver(GnnObserver("uc_hetrogeneous_5_1_sb_1.pkl", try_use_gpu=False), scip_params, pool,
        #                    node_limit=100, score_method="speedup", local=True, n_proc=2),
        #     scip_params
        # ),
        # EcoleSolver(
        #     LiftedObserver(GnnObserver("uc_hetrogeneous_5_1_ub_1_1_local.pkl", try_use_gpu=False), scip_params, pool,
        #                    node_limit=100, score_method="speedup", local=True, n_proc=2),
        #     scip_params
        # ),
        # EcoleSolver(
        #     LiftedObserver(GnnObserver("uc_hetrogeneous_5_1_ub_2_1_local.pkl", try_use_gpu=False), scip_params, pool,
        #                    node_limit=100, score_method="speedup", local=True, n_proc=2),
        #     scip_params
        # ),
    ]


    metrics = ["nnodes", "time", "gap"]

    data = evaluate_solvers(solvers, instances, n_instances, metrics)
    pickle.dump(data, open("data", "wb"))
    # data = pickle.load(open("data", "rb"))

    data.performance_profile(metric="time")
    data.performance_profile(metric="nnodes")