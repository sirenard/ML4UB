import ecole
import pyscipopt


class gapReward:
    def before_reset(self: ecole.reward.NNodes, model: ecole.scip.Model) -> None:
        pass
    def extract(self: ecole.reward.NNodes, model: ecole.scip.Model, done: bool = False) -> float:
        m: pyscipopt.Model = model.as_pyscipopt()
        return m.getGap()

class LpReward:
    def before_reset(self, model: ecole.scip.Model) -> None:
        pass
    def extract(self, model: ecole.scip.Model, done: bool = False):
        m: pyscipopt.Model = model.as_pyscipopt()

        lp_obj = m.getLPObjVal()
        vars = []
        for var in m.getVars():
            #v: pyscipopt.scip.Variable
            val = var.getLPSol()
            vars.append(val)

        return lp_obj, vars