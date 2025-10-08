import argparse

import ecole
import numpy as np
import torch
from boundml.components.gnn_component import GnnBranching

from boundml.evaluation import evaluate_solvers
from boundml.instances import CombinatorialAuctionGenerator

from boundml.components import StrongBranching, AccuracyBranching, EcoleComponent
from boundml.solvers import *
from pyomo.core import components

from lb_component import LiftedBranching
from ucig_generator import UcigGenerator
from utils import get_instances, get_component

if __name__ == '__main__':

    import multiprocess.context as ctx
    ctx._force_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=str)
    parser.add_argument("--n_instances", type=int, default=5)
    parser.add_argument("--timelimit", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--ncpu", type=int, default=24)
    parser.add_argument(
        '-p', '--solver_pair',
        nargs=2,
        action='append',
        required=True,
    )
    args = parser.parse_args()

    seed = args.seed
    start = 0
    if args.instances == "cflfixed":
        seed = 12
        start = args.start_at
    elif args.instances == "uc":
        start = 5000

    instances = get_instances(args.instances, "easy", seed, start)

    scip_params = {
        "timing/clocktype": 1,
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
        components.append(AccuracyBranching(model, expert))



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
