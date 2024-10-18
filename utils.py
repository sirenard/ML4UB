import ecole

import pyscipopt

from Event import MyEvent
from OutputControl import OutputControl
from Separator import MySperator


def get_ecole_model_from_file(path):
    model = pyscipopt.Model()
    model.readProblem(path)
    return ecole.scip.Model.from_file(path)


def configure(model: pyscipopt.Model, branching_policy="vanillafullstrong"):
    # model.setPresolve(PY_SCIP_PARAMSETTING.OFF)
    # model.setSeparating(PY_SCIP_PARAMSETTING.OFF)
    # model.setHeuristics(PY_SCIP_PARAMSETTING.OFF)

    model.setParam("randomization/permutevars", False)
    model.setParam("randomization/lpseed", 1)
    model.setParam("randomization/randomseedshift", 1)
    model.setParam("randomization/permutationseed", 1)
    model.setParam("display/verblevel", 0)

    # model.setParam("separating/maxrounds", 0)
    # model.setParam("presolving/maxrestarts", 0)

    # model.includeEventhdlr(MyEvent(), "myevent", "myevent")

    model.setParam(f"branching/{branching_policy}/priority", 9999999)


def solve(model: pyscipopt.Model):
    model.optimize()
    nnodes = model.getNNodes()
    obj = model.getObjVal()
    return nnodes, obj


def write_to_file(model_path, solution_path, model: pyscipopt.Model):
    output_control = OutputControl()
    output_control.mute()
    model.writeProblem(model_path, trans=True)
    model.writeBestTransSol(solution_path, write_zeros=True)
    output_control.unmute()

