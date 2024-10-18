import tempfile

import ecole
import pyscipopt

from solvers import ClassicSolver


class gapReward:
        def before_reset(self: ecole.reward.NNodes, model: ecole.scip.Model) -> None:
            pass
        def extract(self: ecole.reward.NNodes, model: ecole.scip.Model, done: bool = False) -> float:
            m: pyscipopt.Model = model.as_pyscipopt()
            return m.getGap()


class BaseLineTimeReward:
    def __init__(self):
        self.baseline_time = 0
    def before_reset(self: ecole.reward.NNodes, model: ecole.scip.Model) -> None:
        prob_file = tempfile.NamedTemporaryFile(suffix=".lp", dir="tmp")
        model.as_pyscipopt().writeProblem(prob_file.name)

        solver = ClassicSolver("relpscost", {})
        solver.solve(prob_file.name)
        self.baseline_time = solver["time"]

    def extract(self: ecole.reward.NNodes, model: ecole.scip.Model, done: bool = False) -> float:
        m: pyscipopt.Model = model.as_pyscipopt()
        return (self.baseline_time - m.getTotalTime())/self.baseline_time