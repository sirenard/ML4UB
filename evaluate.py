import pickle

import ecole.instance
import pyscipopt

import linear_model
from evaluation_tools import evaluate_solvers
from model import train
import ranking_model
from observer import LiftedObserver, StrongBranching, GnnObserver, SVMObserver, LinearObserver
from solvers import *

from uc import UcigGenerator


if __name__ == '__main__':
    #instances = UcigGenerator("instances", start_at_instance=5000, days=1)
    #instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000)
    instances = ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500)
    instances.seed(12)


    n_instances = 10

    scip_params = {
        #"limits/nodes":20000,
        # "limits/time": 120,
    }

    scip_params_ub = {
        "limits/nodes": 5000,
    }

    # policy = train("sc_sb_0.6", 0.001, 15, "sc_sb_0.6.pkl")
    #policy = train("ub_0_1_limit_20_alpha2_none", 0.001, 15, "ub_0_1_limit_20_alpha2_none.pkl")
    #policy = train("ub_0_1_limit_20_alpha2_0.1", 0.001, 15, "ub_0_1_limit_20_alpha2_0.1.pkl")
    #rank_model = ranking_model.train("test", "test.pkl")
    #model = linear_model.train("test", "test.pkl")


    def estimator(m: pyscipopt.scip.Model):
        with open("test.pkl", "rb") as f:
            model = pickle.load(f)
        lp_obj = m.getLPObjVal()

        features = []
        for var in m.getVars():
            features.append(var.getLPSol())

        alpha = model.predict([features])[0]
        if alpha < 0:
            alpha=0

        return (1+alpha)*lp_obj

    def estimator2(m: pyscipopt.scip.Model):
        m2 = pyscipopt.scip.Model(sourceModel=m)
        m2.setParam("display/verblevel", 0)
        m2.optimize()
        return m2.getObjVal()




    solvers = [
        # ClassicSolver("vanillafullstrong", scip_params),
        ClassicSolver("relpscost", scip_params),
        ClassicSolver("boundbranch", scip_params),
        # EcoleSolver(GnnObserver("sb_0.5.pkl", try_use_gpu=True), scip_params=scip_params),
        #EcoleSolver(GnnObserver("ub_0_0.6.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("ub_0_speedup_0.6.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("ub_2_0.6_8.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("ub_1_speedup_0.6.pkl", try_use_gpu=True), scip_params),
        # EcoleSolver(GnnObserver("sc_sb_0.6.pkl", try_use_gpu=True), scip_params),

        #EcoleSolver(LinearObserver("test.pkl"), scip_params),
        #EcoleSolver(SVMObserver("test.pkl"), scip_params),

        # EcoleSolver(
        #      LiftedObserver(StrongBranching(), scip_params_ub, pool),
        # ),
        # EcoleSolver(
        #     LiftedObserver(GnnObserver("ub_1_0.6.pkl", try_use_gpu=False), scip_params_ub, pool),
        # ),
    ]


    metrics = ["nnodes", "time", "gap", "obj"]

    data = evaluate_solvers(solvers, instances, n_instances, metrics)
    pickle.dump(data, open("data", "wb"))

    data.performance_profile(metric="time")
    data.performance_profile(metric="nnodes")
