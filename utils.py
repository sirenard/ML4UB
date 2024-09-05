import sys
import tempfile

from pyscipopt.scip import PY_SCIP_PARAMSETTING

from branching import BoundBranching
import ecole

import pyscipopt


class OutputControl:
    def __init__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.trash = open("/dev/null", "w")

    def mute(self):
        sys.stdout = self.trash
        sys.stderr = self.trash

    def unmute(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr

    def __del__(self):
        self.trash.close()


def get_ecole_model_from_file(path):
    model = pyscipopt.Model()
    model.readProblem(path)
    return ecole.scip.Model.from_file(path)



def get_objval_from_existing_model(model: pyscipopt.Model):
    output_control = OutputControl()

    prob_file = tempfile.NamedTemporaryFile(suffix=".lp", dir="tmp")
    output_control.mute()
    model.writeProblem(prob_file.name)

    m2 = pyscipopt.Model()
    m2.setParam("display/verblevel", 0)

    m2.readProblem(prob_file.name)
    m2.optimize()
    output_control.unmute()

    res = m2.getSolObjVal(m2.getBestSol())

    prob_file.close()

    return res

def configure(model: pyscipopt.Model, branching_policy="vanillafullstrong", estimator_function=get_objval_from_existing_model):
    #model.setPresolve(PY_SCIP_PARAMSETTING.OFF)
    #model.setSeparating(PY_SCIP_PARAMSETTING.OFF)
    #model.setHeuristics(PY_SCIP_PARAMSETTING.OFF)

    model.setParam("randomization/permutevars", False)
    model.setParam("randomization/lpseed", 1)
    model.setParam("randomization/randomseedshift", 1)
    model.setParam("randomization/permutationseed", 1)
    model.setParam("display/verblevel", 0)

    #model.setParam("separating/maxrounds", 0)
    #model.setParam("presolving/maxrestarts", 0)


    if branching_policy == "boundbranch":
        branchrule = BoundBranching(estimator_function)
        model.includeBranchrule(
            branchrule=branchrule,
            name="boundbranch", # name of the branching rule
            desc="",           # description of the branching rule
            priority=100000,   # priority: set to this to make it default
            maxdepth=0,       # maximum depth up to which it will be used, or -1 for no restriction
            maxbounddist=1
        )

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

