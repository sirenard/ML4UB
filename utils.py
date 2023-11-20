import ecole

import pyscipopt
from pyscipopt.scip import PY_SCIP_PARAMSETTING


def get_ecole_model_from_file(path):
    model = pyscipopt.Model()
    model.readProblem(path)
    return ecole.scip.Model.from_file(path)


def configure(model: pyscipopt.Model):
    model.setPresolve(PY_SCIP_PARAMSETTING.OFF)
    model.setSeparating(PY_SCIP_PARAMSETTING.OFF)
    model.setHeuristics(PY_SCIP_PARAMSETTING.OFF)

    model.setParam("randomization/permutevars", False)
    model.setParam("randomization/lpseed", 1)
    model.setParam("randomization/randomseedshift", 1)
    model.setParam("randomization/permutationseed", 1)
    model.setParam("display/verblevel", 0)

    model.setParam("branching/vanillafullstrong/priority", 9999999)

def solve(model: pyscipopt.Model):
    model.optimize()
    nnodes = model.getNNodes()
    obj = model.getObjVal()
    return nnodes, obj
