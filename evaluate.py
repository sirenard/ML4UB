import pickle
import ecole.instance
from boundml.solvers import ClassicSolver

from ucig_generator import UcigGenerator



if __name__ == '__main__':
    instances = UcigGenerator("instances", start_at_instance=5000, days=1)

    n_instances = 10

    scip_params = {
        # "estimation/method": "c",
        # "estimation/completiontype": "m",
        "limits/time":120,
        #"limits/nodes": 50,
        # "branching/relpscost/maxreliable": 4,
        # "branching/relpscost/minreliable": 4,
        # "branching/relpscost/usesblocalinfo": True,
    }

    # policy = train("sb_normal_0.6", 0.005, 30, "sb_normal_0.6.pkl")
    # policy = train("sb_extended_0.6", 0.005, 17, "sb_extended_0.6.pkl", n_tree_features=5)
    # policy = train("ca_100_500_ub_1_0.6", 0.005, 20, "ca_100_500_ub_1_0.6.pkl")
    # policy = train("ca_100_500_sb_0.6", 0.005, 20, "ca_100_500_sb_0.6.pkl")
    # policy = train("sc_500_1000_ub_0_1_local", 0.005, 30, "sc_500_1000_ub_0_1_local.pkl")

    # torch.set_num_threads(1)

    # from genetic_branching import toolbox
    # pop = pickle.load(open("best_gen2", "rb"))

    # f = toolbox.compile(expr=pop)

    # pop = pickle.load(open("best_gen_uc_5", "rb"))

    # f2 = toolbox.compile(expr=pop)

    # acc_observers = AccuracyObserver(StrongBranching(), GnnObserver("uc_5_1_sb_0.6.pkl", try_use_gpu=False))

    solvers = [
        # EcoleSolver(StrongBranching(), scip_params),
        ClassicSolver("relpscost", scip_params),
        # ClassicSolver("pscost", scip_params),
        # EcoleSolver(FunctorObserver(f), scip_params),
        # EcoleSolver(TestObserver(), scip_params),
        # EcoleSolver(ReliabiltyBranchingFNT(eta=8), scip_params),


        # EcoleSolver(GnnObserver("ca_100_500_sb_0.6.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("ca_100_500_ub_1_0.6.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("sb_extended_0.6.pkl", try_use_gpu=True, feature_observer=NodeBipartiteExtended(),
        #                         n_tree_features=5), scip_params),

        # EcoleSolver(GnnObserver("uc_hetrogeneous_5_1_ub_2_0.6_local.pkl", try_use_gpu=True), scip_params),

        # EcoleSolver(GnnObserver("uc_5_1_sb_0.6.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_5_1_ub_2_1_local.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_5_1_ub_1_0.6_global.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_5_1_ub_1_0.6_local.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_5_1_ub_2_0.6_local.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_5_1_ub_1_0.6_local.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_5_1_ub_2_0.6_local.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_5_1_ub_3_0.6_local.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_5_1_ub_1_0.6_global.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_5_1_ub_2_0.6_global.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("uc_5_1_ub_1_0.6_local.pkl", try_use_gpu=True), scip_params),

        # EcoleSolver(GnnObserver("uc_5_1_ub_3_0.6_local.pkl", try_use_gpu=True), scip_params),

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
        #     LiftedObserver(GnnObserver("ca_100_500_ub_1_0.6.pkl", try_use_gpu=False), scip_params,
        #                    node_limit=100, score_method="speedup", local=True, n_proc=20),
        #     scip_params
        # ),
    ]


    metrics = ["nnodes", "time", "gap"]

    data = evaluate_solvers(solvers, instances, n_instances, metrics)
    pickle.dump(data, open("data", "wb"))
    # data = pickle.load(open("data", "rb"))

    # data.performance_profile(metric="time")
    # data.performance_profile(metric="nnodes")