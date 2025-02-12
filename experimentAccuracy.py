import torch

from evaluation_tools import evaluate_solvers
from observer import LiftedObserver, StrongBranching, GnnObserver, AccuracyObserver
from solvers import *

from ucig_generator import UcigGenerator

if __name__ == '__main__':
    instances = UcigGenerator("instances", start_at_instance=5000, days=1)

    n_instances = 50

    scip_params = {
        "estimation/method": "c",
        "estimation/completiontype": "m",
    }

    torch.set_num_threads(1)

    acc_observers = [
        AccuracyObserver(
            StrongBranching(),
            GnnObserver("uc_5_1_sb_0.6.pkl", try_use_gpu=False)
        ),
        AccuracyObserver(
            LiftedObserver(GnnObserver("uc_5_1_ub_1_0.6_local.pkl", try_use_gpu=False), scip_params, node_limit=100,
                           local=True, n_proc=24),
            GnnObserver("uc_5_1_ub_1_0.6_local.pkl", try_use_gpu=False),
        ),
        AccuracyObserver(
            LiftedObserver(GnnObserver("uc_5_1_ub_2_0.6_local.pkl", try_use_gpu=False), scip_params, node_limit=100,
                           local=True, n_proc=24),
            GnnObserver("uc_5_1_ub_2_0.6_local.pkl", try_use_gpu=False),
        ),
        AccuracyObserver(
            LiftedObserver(GnnObserver("uc_5_1_ub_3_0.6_local.pkl", try_use_gpu=False), scip_params, node_limit=100,
                           local=True, n_proc=24),
            GnnObserver("uc_5_1_ub_3_0.6_local.pkl", try_use_gpu=False),
        ),
        AccuracyObserver(
            LiftedObserver(GnnObserver("uc_5_1_ub_4_0.6_local.pkl", try_use_gpu=False), scip_params, node_limit=100,
                           local=True, n_proc=24),
            GnnObserver("uc_5_1_ub_4_0.6_local.pkl", try_use_gpu=False),
        ),

    ]

    solvers = [
        EcoleSolver(
            GnnObserver("uc_5_1_sb_0.6.pkl", try_use_gpu=False), scip_params,
            additional_observers=[acc_observers[0]]),
        EcoleSolver(GnnObserver("uc_5_1_ub_1_0.6_local.pkl", try_use_gpu=False), scip_params,
                    additional_observers=[acc_observers[1]]),
        EcoleSolver(GnnObserver("uc_5_1_ub_2_0.6_local.pkl", try_use_gpu=False), scip_params,
                    additional_observers=[acc_observers[2]]),
        EcoleSolver(GnnObserver("uc_5_1_ub_3_0.6_local.pkl", try_use_gpu=False), scip_params,
                    additional_observers=[acc_observers[3]]),
        EcoleSolver(GnnObserver("uc_5_1_ub_4_0.6_local.pkl", try_use_gpu=False), scip_params,
                    additional_observers=[acc_observers[4]]),
    ]

    metrics = ["nnodes", "time", "gap"]

    evaluate_solvers(solvers, instances, n_instances, metrics)

    print()
    for observer in acc_observers:
        print(f"{observer} {observer.get_observations().size}")
        for n in [1, 5, 10]:
            print(f"Acc@{n}: {100 * np.average(observer.get_observations() <= n)}")
        print()
