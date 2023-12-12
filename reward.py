import ecole
import pyscipopt


class gapReward:
        def before_reset(self: ecole.reward.NNodes, model: ecole.scip.Model) -> None:
            pass
        def extract(self: ecole.reward.NNodes, model: ecole.scip.Model, done: bool = False) -> float:
            m: pyscipopt.Model = model.as_pyscipopt()
            return m.getGap()