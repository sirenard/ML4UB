import argparse
import pickle

from boundml.evaluation import evaluate_solvers
from boundml.solvers import DefaultScipSolver, ModularSolver

from lb_component import LiftedBranching

import torch

from utils import get_instances, get_component

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=str)
    parser.add_argument("--difficulty", type=str, choices=['easy', 'medium', 'hard'])
    parser.add_argument("--out", type=str, default="out.pkl")
    parser.add_argument("--n_instances", type=int, default=5)
    parser.add_argument("--solvers", type=str, nargs="*")
    parser.add_argument("--solvers", type=str, nargs="*")
    parser.add_argument("--ncpu", type=int, required=True)
    parser.add_argument("--timelimit", type=int, default=600)
    parser.add_argument("--usegpu", action="store_true", default=False)
    parser.add_argument("--ntorchthreads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1212)

    args = parser.parse_args()

    seed = args.seed
    start = 0
    if args.instances == "cflfixed":
        seed = 12
        start = args.seed
    elif args.instances == "uc":
        start = 5000

    instances = get_instances(args.instances, args.difficulty, seed, start)

    if args.ntorchthreads is not None:
        print("using ntorch threads", args.ntorchthreads)
        torch.set_num_threads(args.ntorchthreads)

    scip_params = {
        "timing/clocktype": 1,
        "limits/time":args.timelimit,
        'estimation/method': 'c',
        "estimation/completiontype": "m",
        'randomization/permutevars': True,
        'randomization/permutationseed': args.seed,
        'randomization/randomseedshift': args.seed,
        'separating/maxrounds': 0,
        'presolving/maxrestarts': 0,
    }

    import multiprocess.context as ctx

    ctx._force_start_method('spawn')

    solvers = []

    if args.solvers is not None:
        for s in args.solvers:
            solver = DefaultScipSolver(s, scip_params)
            solvers.append(solver)

    use_cpus_for_lb = False
    if args.gnn_agents is not None:
        for agent in args.gnn_agents:
            component = get_component(agent, args.usegpu, args.ncpu)
            if type(component) is LiftedBranching:
                use_cpus_for_lb = True

            solver = ModularSolver(
                component,
                scip_params=scip_params,
            )

            solvers.append(solver)



    data = evaluate_solvers(
        solvers,
        instances,
        args.n_instances,
        ["nnodes", "time", "gap"],
        n_cpu = args.ncpu if not use_cpus_for_lb else 1,
    )

    pickle.dump(data, open(args.out, "wb"))
