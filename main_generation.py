import ecole.instance

from datasetGenerator import DatasetGenerator
from mpi import MpiPool
from observer import LiftedObserver, StrongBranching, GnnObserver
from solvers import *

from uc import UcigGenerator
import argparse


def main(pool: MpiPool = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_prefix", type=str, default="")
    parser.add_argument("--start_at", type=int, default=0)
    args = parser.parse_args()

    # instances = UcigGenerator("uc_hetrogeneous_5_1", start_at_instance=args.start_at+1)
    instances = ecole.instance.CapacitatedFacilityLocationGenerator()
    instances.seed(args.start_at)

    expert_probability = 1

    scip_params = {
        "limits/nodes": 1000,
    }



    generator = DatasetGenerator(
        instances,
        # expert_observer=LiftedObserver(observer=StrongBranching(), scip_params=scip_params, pool=pool, score_method="speedup", node_limit=500),
        # expert_observer=LiftedObserver(observer=GnnObserver("sc_sb_0.05.pkl", try_use_gpu=False), scip_params=scip_params_ub, pool=pool, score_method="speedup"),
        expert_observer=StrongBranching(),
        exploration_observer=ecole.observation.Pseudocosts(),
        expert_probability=expert_probability,
        #state_observer=ecole.observation.Khalil2016(),
        scip_params=scip_params,
        )
    generator.generate(f"cfltest_sb_{expert_probability}", max_instances=100, sample_prefix=args.sample_prefix)



if __name__ == "__main__":

    t = time.time()
    utils.mpi_start(main)
    t = time.time() - t
    print(f"Runned in {round(t, 2)} s")
