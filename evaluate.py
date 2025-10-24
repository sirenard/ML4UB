import argparse
import pickle

from boundml.evaluation import evaluate_solvers
from boundml.solvers import DefaultScipSolver, ModularSolver

from lb_component import LiftedBranching

import torch

from utils import get_instances, get_component

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solve instances using different solvers to compare them.")
    parser.add_argument("--instances", type=str, choices=["ca", "cfl", "sc", "mis", "uc"])
    parser.add_argument("--difficulty", type=str, choices=['easy', 'medium', 'hard'])
    parser.add_argument("--out", type=str, default="out.pkl", help="Where to save all the data to compute reports later")
    parser.add_argument("--n_instances", type=int, default=5, help="Number of instances to solve")
    parser.add_argument("--solvers", type=str, nargs="*", help=
        """"Which custom solvers to use. 
        sb -> Strong Branching
        lb_sb -> Lifted Branching of strong branching
        lb_path -> Lifted branching, using GNN branching using model in `path`
        path -> GNN branching using model in `path`"""
    )
    parser.add_argument("--classical_solvers", type=str, nargs="*", help="WHich default SCIP's strategies to used (relpscost, pscost, ...)")
    parser.add_argument("--ncpu", type=int, required=True, help="Number of CPU cores to use. If no LB solver is used, ncpu instances are solved in parallel. If a LB solver is used, ncpu are used for computing LB scores")
    parser.add_argument("--timelimit", type=int, default=600, help="Time limit in seconds for each instance")
    parser.add_argument("--usegpu", action="store_true", default=False, help="Should GNN strategies use GPU")
    parser.add_argument("--ntorchthreads", type=int, default=None, help="Number of torch threads to use")
    parser.add_argument("--seed", type=int, default=1212)

    args = parser.parse_args()

    seed = args.seed
    start = 0
    if args.instances == "cflfixed":
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

    if args.classical_solvers is not None:
        for s in args.classical_solvers:
            solver = DefaultScipSolver(s, scip_params)
            solvers.append(solver)

    use_cpus_for_lb = False
    if args.solvers is not None:
        for agent in args.solvers:
            component = get_component(agent, args.usegpu, args.ncpu)
            if type(component) is LiftedBranching:
                use_cpus_for_lb = True

            solver = ModularSolver(
                component,
                scip_params=scip_params,
            )

            solvers.append(solver)

    print("LB solvers: ", use_cpus_for_lb)


    ncpu = args.ncpu if not use_cpus_for_lb else 1
    print(f"Evaluate using {ncpu} processes")

    data = evaluate_solvers(
        solvers,
        instances,
        args.n_instances,
        ["nnodes", "time", "gap"],
        n_cpu = ncpu,
    )

    pickle.dump(data, open(args.out, "wb"))
