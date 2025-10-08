
import argparse

import ecole
import torch
from boundml.components import EcoleComponent, StrongBranching
from boundml.components.gnn_component import GnnBranching
from boundml.instances import CombinatorialAuctionGenerator, CapacitatedFacilityLocationGenerator, SetCoverGenerator, \
    IndependentSetGenerator
from boundml.ml import BranchingDatasetGenerator

from lb_component import LiftedBranching
from utils import get_instances

# from ucig_generator import UcigGenerator

if __name__ == '__main__':
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_prefix", type=str, default="")
    parser.add_argument("--start_at", type=int, default=0)
    parser.add_argument("--instances", type=str)
    parser.add_argument('--lsb', action='store_true', default=False)
    parser.add_argument("--expert", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--ncpu", type=int, required=True)
    parser.add_argument("--nodelimit", type=int, default=100)
    parser.add_argument("--maxinstances", type=int, default=100)
    parser.add_argument("--local", type=bool, default=True)

    args = parser.parse_args()

    seed = args.start_at
    scip_params = {
        "estimation/method": "c",
        "estimation/completiontype": "m",
        "limits/nodes": 500,
        'randomization/permutevars': True,
        'randomization/permutationseed': seed,
        'randomization/randomseedshift': seed,
        'separating/maxrounds': 0,
        'presolving/maxrestarts': 0,
    }

    start=0
    if args.instances == "cflfixed":
        seed = 12
        start=args.start_at

    instances = get_instances(args.instances, "easy", seed, start)

    state_component = EcoleComponent(ecole.observation.NodeBipartite())

    if not args.lsb:
        if args.expert == "sb":
            strategy = StrongBranching(True, False, True, True)

        else:
            strategy = GnnBranching(
                f"{args.expert}.pkl",
                feature_component = state_component,
                try_use_gpu=False
            )

        expert = LiftedBranching(
            expert=strategy,
            scip_params=scip_params,
            score_method="speedup",
            node_limit=args.nodelimit,
            local=args.local,
            n_proc=args.ncpu
        )
    else:
        expert = StrongBranching(False, True, True, True)
        print("Use SB as expert")

    expert_probability = 0.6

    generator = BranchingDatasetGenerator(
        instances,
        expert_strategy=expert,
        state_component_observer=state_component,
        expert_probability=expert_probability,
        scip_params=scip_params,
    )

    generator.generate(f"{args.out}_{expert_probability}", max_instances=args.maxinstances, sample_prefix=args.sample_prefix)


