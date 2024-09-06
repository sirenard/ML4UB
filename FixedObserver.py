import time

import numpy as np
import pyscipopt

import utils
from observer import Observer
from solvers import ClassicSolver


def get_pseudocost_score(instance_path):
    output_control = utils.OutputControl()
    output_control.mute()

    solver = ClassicSolver("relpscost")
    solver.solve(instance_path)

    output_control.unmute()

    m: pyscipopt.Model = solver.model

    scores = {}
    sol = m.getBestSol()
    for var in m.getVars():
        val = sol[var]
        var = m.getTransformedVar(var)
        ps_score = m.getVarPseudocostScore(var, val)
        scores[var.name] = ps_score

    return scores

class FixedObserver(Observer):
    def __init__(self, scip_params, score_function=get_pseudocost_score):
        super().__init__()
        self.scip_params = scip_params
        self.scores = {}
        self.score_function = score_function

    def before_reset(self, model):
        pass

    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = m.getLPBranchCands()
        n_vars = int(m.getNVars())

        res = np.zeros(n_vars)
        res[:] = np.nan

        for var in candidates:
            var: pyscipopt.Variable
            prob_indexes = var.getCol().getLPPos()
            s = self.scores[var.name]
            res[prob_indexes] = s

        return res


    def reset(self, instance_path, seed=None):
        super().reset(instance_path, seed)
        self.scores = self.score_function(instance_path)



    def __str__(self):
        return "Fixed"


