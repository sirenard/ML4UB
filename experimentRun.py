import argparse
import pickle

import torch

from evaluation_tools import evaluate_solvers
from observer import LiftedObserver, StrongBranching, GnnObserver, AccuracyObserver
from solvers import *

from uc import UcigGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--start_at", type=int, default=0)
    parser.add_argument("--observers",nargs='+', type=str)
    parser.add_argument("--time_limit", type=int, default=9999999)
    parser.add_argument("--n_instances", type=int, default=10)
    parser.add_argument("--out", type=str, default="data.pkl")
    args = parser.parse_args()


    instances = UcigGenerator("instances", start_at_instance=args.start_at, days=args.days)

    n_instances = args.n_instances

    scip_params = {
        "estimation/method": "c",
        "estimation/completiontype": "m",
        "limits/time": args.time_limit,
    }

    print(scip_params)

    torch.set_num_threads(1)

    solvers = []

    for arg in args.observers:
        if arg == "sb":
            solvers.append(
                EcoleSolver(StrongBranching(), scip_params),
            )
        elif arg == "relpscost":
            solvers.append(
                ClassicSolver("relpscost", scip_params),
            )
        elif arg == "lb_sb":
            solvers.append(
                EcoleSolver(
                    LiftedObserver(StrongBranching(), scip_params, node_limit=100, score_method="speedup", local=True, n_proc=24),
                    scip_params
                )
            )
        elif "lb" in arg:
            name = "_".join(arg.split("_")[1:])
            solvers.append(
                EcoleSolver(
                    LiftedObserver(
                        GnnObserver(name, try_use_gpu=False),
                        scip_params,
                        node_limit=100,
                        score_method="speedup",
                        local=True,
                        n_proc=24),
                    scip_params
                )
            )
        else:
            solvers.append(
                EcoleSolver(GnnObserver(arg, try_use_gpu=False), scip_params),
            )

    metrics = ["nnodes", "time", "gap"]

    data = evaluate_solvers(solvers, instances, n_instances, metrics)
    pickle.dump(data, open(args.out, "wb"))


