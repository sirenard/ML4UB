import tempfile

import numpy as np
import pyscipopt
from pyscipopt import Branchrule, Eventhdlr


class BoundBranching(pyscipopt.scip.Branchrule):
    def __init__(self, bound_estimator=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0
        self.bound_estimator = bound_estimator


    def branchexeclp(self, allowaddcons):
        self.count += 1
        bound_variable = [None, None]
        for var in self.model.getVars():
            if "choseBound" in var.name:
                if "_1" in var.name:
                    bound_variable[0] = self.model.getTransformedVar(var)
                elif "_2" in var.name:
                    bound_variable[1] = self.model.getTransformedVar(var)

        x1, x2 = bound_variable
        # Child 1: child containing (hopefully) the optimal solution
        child_1: pyscipopt.scip.Node = self.model.createChild(999999, 0)
        self.model.chgVarLbNode(child_1, x1, 1)
        self.model.chgVarLbNode(child_1, x2, 1)


        #child_2: pyscipopt.scip.Node = self.model.createChild(-1000, np.Inf)
        #self.model.chgVarLbNode(child_2, var, (1+alpha)*upper_bound)

        self.model.setParam(f"branching/relpscost/priority", 9999998)

        return {"result": pyscipopt.SCIP_RESULT.BRANCHED}

