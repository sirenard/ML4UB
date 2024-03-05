import ecole.instance

from datasetGenerator import DatasetGenerator
from mpi import MpiPool
from observer import LiftedObserver, StrongBranching, GnnObserver
from solvers import *

from uc import UcigGenerator
import argparse


def main(pool: MpiPool = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_prefix", type=str, default="", required=False)
    parser.add_argument("--start_at", type=int, default=0, required=False)
    args = parser.parse_args()

    instances = UcigGenerator("instances", start_at_instance=args.start_at)

    expert_probability = 1

    scip_params = {
        "limits/nodes": 200,
    }

    scip_params_ub = {
        "limits/nodes": 500
    }


    generator = DatasetGenerator(
        instances,
        #expert_observer=LiftedObserver(observer=StrongBranching(), scip_params=scip_params_ub, pool=pool, score_method="n"),
        #expert_observer=LiftedObserver(observer=GnnObserver("sb_0.5.pkl", try_use_gpu=False), scip_params=scip_params_ub, pool=pool),
        expert_observer=StrongBranching(),
        exploration_observer=ecole.observation.Pseudocosts(),
        expert_probability=expert_probability,
        #state_observer=ecole.observation.Khalil2016(),
        scip_params=scip_params,
        )
    #generator.generate(f"ub_0_{expert_probability}", max_instances=2, sample_prefix=args.sample_prefix)
    generator.generate(f"sb_test", max_instances=250, sample_prefix=args.sample_prefix)



if __name__ == "__main__":

    t = time.time()
    utils.mpi_start(main)
    t = time.time() - t
    print(f"Runned in {round(t, 2)} s")
