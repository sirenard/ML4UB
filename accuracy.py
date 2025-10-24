import argparse

import torch

from boundml.evaluation import evaluate_solvers

from boundml.components import AccuracyBranching
from boundml.solvers import *

from utils import get_instances, get_component

if __name__ == '__main__':

    import multiprocess.context as ctx
    ctx._force_start_method('spawn')

    parser = argparse.ArgumentParser(description=
        """
        Solve the instances using the model strategy.
        At each decision, compute expert scores and store how the model's decision is ranked according to expert scores.
        After solving all the instances, report the Acc@1 Acc@5 Acc@10 in percent.
        """
    )
    parser.add_argument("--instances", type=str, choices=["ca", "cfl", "sc", "mis", "uc"])
    parser.add_argument("--n_instances", type=int, default=5)
    parser.add_argument("--timelimit", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--ncpu", type=int, default=24, help="Number of CPUs to use for LB components")
    parser.add_argument(
        '-p', '--solver_pair',
        nargs=2,
        action='append',
        required=True,
        help="""Pair of solvers: expert, model. The format is the same one as used for evaluation: 
        sb -> Strong Branching
        lb_sb -> Lifted Branching of strong branching
        lb_path -> Lifted branching, using GNN branching using model in `path`
        path -> GNN branching using model in `path`
        """
    )
    args = parser.parse_args()

    seed = args.seed
    start = 0
    if args.instances == "cflfixed":
        seed = 12
        start = seed
    elif args.instances == "uc":
        start = 5000

    instances = get_instances(args.instances, "easy", seed, start)

    scip_params = {
        # "timing/clocktype": 1,
        "limits/time": args.timelimit,
        'estimation/method': 'c',
        "estimation/completiontype": "m",
        'randomization/permutevars': True,
        'randomization/permutationseed': args.seed,
        'randomization/randomseedshift': args.seed,
        'separating/maxrounds': 0,
        'presolving/maxrestarts': 0,
    }

    torch.set_num_threads(1)

    print(args.solver_pair)
    components = []
    for model_name in args.solver_pair:
        expert = get_component(model_name[0], usegpu=False, ncpu=args.ncpu)
        model = get_component(model_name[1], usegpu=False, ncpu=args.ncpu)
        components.append(AccuracyBranching(expert, model))



    solvers = [
        ModularSolver(
            component,
            scip_params = scip_params,
        ) for component in components
    ]

    metrics = ["nnodes", "time", "gap"]

    evaluate_solvers(solvers, instances, args.n_instances, metrics, n_cpu=1)

    print()
    for component in components:
        print(f"{component} {component.get_observations().size}")
        for n in [1, 5, 10]:
            print(f"Acc@{n}: {100 * component.get_accuracy(n)}")
        print()
