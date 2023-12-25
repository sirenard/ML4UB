import sys
from mpi import MpiPool
from mpi4py import MPI
import ecole

import pyscipopt
from pyscipopt.scip import PY_SCIP_PARAMSETTING


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


def configure(model: pyscipopt.Model, branching_policy="vanillafullstrong"):
    #model.setPresolve(PY_SCIP_PARAMSETTING.OFF)
    #model.setSeparating(PY_SCIP_PARAMSETTING.OFF)
    #model.setHeuristics(PY_SCIP_PARAMSETTING.OFF)

    model.setParam("randomization/permutevars", False)
    model.setParam("randomization/lpseed", 1)
    model.setParam("randomization/randomseedshift", 1)
    model.setParam("randomization/permutationseed", 1)
    model.setParam("display/verblevel", 0)

    model.setParam("separating/maxrounds", 0)
    model.setParam("presolving/maxrestarts", 0)

    model.setParam(f"branching/{branching_policy}/priority", 9999999)


def solve(model: pyscipopt.Model):
    model.optimize()
    nnodes = model.getNNodes()
    obj = model.getObjVal()
    return nnodes, obj


def mpi_start(f, *args, **kwargs):
    n_mpi = MPI.COMM_WORLD.Get_size()
    with MpiPool() as pool:
        if pool is not None:
            print(f"{n_mpi} workers")
            f(pool, *args, **kwargs)
