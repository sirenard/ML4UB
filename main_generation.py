
import argparse

import ecole
import torch
from boundml.components import EcoleComponent, StrongBranching
from boundml.components.gnn_component import GnnBranching
from boundml.instances import CombinatorialAuctionGenerator, CapacitatedFacilityLocationGenerator, SetCoverGenerator, \
    IndependentSetGenerator
from boundml.ml import BranchingDatasetGenerator

from lb_component import LiftedBranching
# from ucig_generator import UcigGenerator

if __name__ == '__main__':
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_prefix", type=str, default="")
    parser.add_argument("--start_at", type=int, default=0)
    parser.add_argument("--instances", type=str)
    parser.add_argument("--expert", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--ncpu", type=int, required=True)
    parser.add_argument("--nodelimit", type=int, default=100)
    parser.add_argument("--maxinstances", type=int, default=100)
    parser.add_argument("--local", type=bool, default=True)

    args = parser.parse_args()

    match args.instances:
        # case "uc":
        #     instances = UcigGenerator("instances", start_at_instance=args.start_at, days=1)
        case "ca":
            instances = CombinatorialAuctionGenerator(100, 500)
        case "cfl":
            instances = CapacitatedFacilityLocationGenerator(100)
        case "sc":
            instances = SetCoverGenerator(500, 1000)
        case "mis":
            instances = IndependentSetGenerator(500)

        case _:
            raise NotImplementedError

    if args.instances != "uc":
        instances.seed(args.start_at)

    state_component = EcoleComponent(ecole.observation.NodeBipartite())

    if args.expert == "sb":
        expert = StrongBranching(True, False, True, True)

    else:
        expert = GnnBranching(
            f"{args.expert}.pkl",
            feature_component = state_component,
            try_use_gpu=False
        )

    expert_probability = 0.6

    scip_params = {
        "estimation/method": "c",
        "estimation/completiontype": "m",
        "limits/nodes": 200,
    }


    generator = BranchingDatasetGenerator(
        instances,
        expert_strategy=LiftedBranching(
            expert=expert,
            scip_params=scip_params,
            score_method="speedup",
            node_limit=args.nodelimit,
            local=args.local,
            n_proc=args.ncpu
        ),
        state_component_observer=state_component,
        expert_probability=expert_probability,
        scip_params=scip_params,
    )

    generator.generate(f"{args.out}_{expert_probability}", max_instances=args.maxinstances, sample_prefix=args.sample_prefix)


