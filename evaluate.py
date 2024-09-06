import pickle

import ecole.instance
import numpy as np

import torch

from AcsObserver import acs_score, sb_score
from FixedObserver import FixedObserver
from model import train
from evaluation_tools import evaluate_solvers
from observer import LiftedObserver, StrongBranching, GnnObserver, SVMObserver, LinearObserver, EntropyObserver, \
    ConditionalObservers, RandomObserver, NodeBipartiteExtended, AccuracyObserver
from solvers import *

from uc import UcigGenerator



if __name__ == '__main__':
    instances = ecole.instance.CombinatorialAuctionGenerator( n_items=100, n_bids=500)
    #instances = ecole.instance.SetCoverGenerator(n_rows=1000, n_cols=1000)
    #instances = UcigGenerator("instances")
    instances.seed(12)

    n_instances = 10

    scip_params = {
        "estimation/method": "c",
        "estimation/completiontype": "m",
        "nodeselection/dfs/stdpriority": 1073741823,
        # "limits/nodes":1000,
        "limits/time": 300,
    }

    solvers = [
        ClassicSolver("relpscost", scip_params),
        #ClassicSolver("pscost", scip_params),
        EcoleSolver(FixedObserver(scip_params), scip_params),
        EcoleSolver(FixedObserver(scip_params, score_function=acs_score), scip_params),
    ]


    metrics = ["nnodes", "time", "gap"]

    data = evaluate_solvers(solvers, instances, n_instances, metrics)
    pickle.dump(data, open("data", "wb"))
    # data = pickle.load(open("data", "rb"))

    data.performance_profile(metric="time")
    data.performance_profile(metric="nnodes")