import pickle

import ecole.instance
import numpy as np

import torch

from ACS import ACSObserer
from AcsObserver import sb_score, _acs_score, get_acs_score_fct
from FixedObserver import FixedObserver
from Mcts import mcts_score, brut_force_score
from model import train
from evaluation_tools import evaluate_solvers
from observer import LiftedObserver, StrongBranching, GnnObserver, SVMObserver, LinearObserver, EntropyObserver, \
    ConditionalObservers, RandomObserver, NodeBipartiteExtended, AccuracyObserver
from solvers import *

from uc import UcigGenerator



if __name__ == '__main__':
    instances = ecole.instance.CombinatorialAuctionGenerator( n_items=300, n_bids=1500)
    #instances = ecole.instance.SetCoverGenerator(n_rows=1000, n_cols=1000)
    #instances = UcigGenerator("instances")
    instances.seed(13)

    n_instances = 10

    scip_params = {
        "estimation/method": "c",
        "estimation/completiontype": "m",
        # "limits/nodes":1000,
        "limits/time": 300,
    }

    solvers = [
        #ClassicSolver("vanillafullstrong", scip_params),
        ClassicSolver("relpscost", scip_params),
        # EcoleSolver(StrongBranching(), scip_params),
        # EcoleSolver(
        #     ACSObserer(
        #         alpha=1.525,
        #         beta=5,
        #         evaporation_rate=0.1,
        #         pheromone_deposit=1000,
        #         num_iterations=3
        #     ), scip_params)

        ClassicSolver("pscost", scip_params),
        EcoleSolver(FixedObserver(scip_params), scip_params),
        # EcoleSolver(FixedObserver(scip_params, score_function=get_acs_score_fct(alpha=0.1, beta=4, evaporation_rate=0.5)), scip_params),
        #EcoleSolver(FixedObserver(scip_params, score_function=brut_force_score), scip_params),
    ]


    metrics = ["nnodes", "time", "gap"]

    data = evaluate_solvers(solvers, instances, n_instances, metrics)
    pickle.dump(data, open("data", "wb"))
    # data = pickle.load(open("data", "rb"))

    data.performance_profile(metric="time")
    data.performance_profile(metric="nnodes")