import ecole.instance

from boundml.dataset_generator import DatasetGenerator
from observer import LiftedObserver
from boundml.observers import StrongBranching
from boundml.solvers import *

from ucig_generator import UcigGenerator
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_prefix", type=str, default="")
    parser.add_argument("--start_at", type=int, default=0)
    parser.add_argument("--instances", type=str)
    parser.add_argument("--expert", type=str)
    parser.add_argument("--out", type=str)

    args = parser.parse_args()

    match args.instances:
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

    #instances = UcigGenerator("instances", tart_at_instance=args.start_at+1)
    instances.seed(args.start_at)


    if args.expert == "sb":
        expert = StrongBranching()
    else:
        expert = GnnObserver("uc_5_1_ub_2_0.6_local.pkl", try_use_gpu=False)

    expert_probability = 0.6

    scip_params = {
        "estimation/method": "c",
        "estimation/completiontype": "m",
        "limits/nodes": 200,
    }



    generator = DatasetGenerator(
        instances,
        expert_observer=LiftedObserver(
            observer=expert,
            scip_params=scip_params,
            score_method="speedup",
            node_limit=100,
            local=True,
            n_proc=24
        ),
        exploration_observer=ecole.observation.Pseudocosts(),
        expert_probability=expert_probability,
        scip_params=scip_params,
        )

    generator.generate(f"{args.out}_{expert_probability}", max_instances=100, sample_prefix=args.sample_prefix)


