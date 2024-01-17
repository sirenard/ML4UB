import ecole
import numpy as np
import pyscipopt
import time

import utils
from bpp import BinpackingGenerator
from environments import HistoryBranchingEnvironment
from mpi import MpiPool
from observer import UnrealisticObserver, ConditionalObservers, DistObserver, GnnObserver, LiftedObserver


class Solver:
    def __init__(self):
        self.model: pyscipopt.Model = None

    def solve(self, path: str):
        raise NotImplementedError

    def configure(self):
        return

    def __getitem__(self, item):
        match item:
            case "time":
                return self.model.getSolvingTime()
            case "nnodes":
                return self.model.getNNodes()
            case "obj":
                return self.model.getObjVal()
            case "gap":
                return self.model.getGap()
            case "sol":
                return self.model.getBestSol()
            case _:
                raise KeyError


class EcoleSolver(Solver):

    def __init__(self, env: ecole.environment, before_start_callbacks=[], before_action_callbacks=[]):
        super().__init__()
        self.env: ecole.environment = env
        self.before_start_callbacks = before_start_callbacks
        self.before_action_callbacks = before_action_callbacks

    def set_before_action_callbacks(self, c):
        self.before_action_callbacks = c

    def set_before_start_callbacks(self, c):
        self.before_start_callbacks = c

    def solve(self, path: str):
        m = pyscipopt.Model()
        m.setParam("display/verblevel", 0)
        m.readProblem(path)
        instance: ecole.scip.Model = ecole.scip.Model.from_pyscipopt(m)
        self.before_start(path)
        self.env.seed(0)

        t = time.time()
        observation, action_set, _, done, info = self.env.reset(instance)
        self.model = instance.as_pyscipopt()

        nb_nodes = info["nb_nodes"]
        while not done:
            action = self.get_action(action_set, observation)
            self.before_action(action, action_set, observation)
            observation, action_set, _, done, info = self.env.step(action)
            self.model = self.env.model.as_pyscipopt()
            nb_nodes += info["nb_nodes"]
            if not done:
                done = self.after_action()

        t = time.time() - t

        self.model = self.env.model.as_pyscipopt()

        return nb_nodes, t

    def before_start(self, path):
        return [f(path) for f in self.before_start_callbacks]

    def before_action(self, action, action_set, observation):
        return [f(action, action_set, observation) for f in self.before_action_callbacks]

    def after_action(self):
        return False

    def get_action(self, action_set, observation) -> int:
        scores = observation[0]
        action_index = scores[action_set].argmax()
        action = action_set[action_index]
        return action


class UBSolver(EcoleSolver):
    def __init__(self, n_repetitions=20, scip_params_env={}, scip_params={}, pool: MpiPool = None,
                 expert_probability=1.0,
                 local=False,
                 additional_observers=[]):
        self.observer = UnrealisticObserver(
            n_repetitions,
            scip_params=scip_params,
            pool=pool,
            local=local,
            expert_probability=expert_probability,
        )

        env = HistoryBranchingEnvironment(
            observation_function=(
                self.observer,
                *additional_observers
            ),
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "time": ecole.reward.SolvingTime(),
            },
            scip_params=scip_params_env,
        )
        self.count = 0
        super().__init__(env)

    def before_action(self, action, action_set, observation):
        super().before_action(action, action_set, observation)
        self.observer.step(action)
        self.count += 1

    def before_start(self, path):
        super().before_start(path)
        self.count = 0
        self.observer.reset(path, 0)

    def __str__(self):
        return "UB"


class SBSolver(EcoleSolver):
    def __init__(self, scip_params={}, additional_observers=[]):
        env = HistoryBranchingEnvironment(
            observation_function=(
                ecole.observation.StrongBranchingScores(),
                *additional_observers
            ),
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "time": ecole.reward.SolvingTime(),
            },
            scip_params=scip_params,
        )

        self.rep = f"SB"
        super().__init__(env)

    def __repr__(self):
        return self.rep

class LiftedSolver(EcoleSolver):
    def __init__(self, scip_params={}, *args, **kargs):
        self.observer = LiftedObserver(*args, **kargs)
        env = HistoryBranchingEnvironment(
            observation_function=(
                self.observer,
            ),
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "time": ecole.reward.SolvingTime(),
            },
            scip_params=scip_params,
        )

        self.rep = f"LB"
        super().__init__(env)

    def before_start(self, path):
        self.observer.reset(path)

    def __repr__(self):
        return self.rep



class RandomSolver(EcoleSolver):
    def __init__(self, scip_params={}, additional_observers=[], seed=None):
        env = HistoryBranchingEnvironment(
            observation_function=additional_observers,
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "time": ecole.reward.SolvingTime(),
            },
            scip_params=scip_params,
        )
        super().__init__(env)
        self.rng = np.random.default_rng(seed)

    def get_action(self, action_set, observation) -> int:
        action = self.rng.choice(action_set)
        return action

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

class DistSolver(EcoleSolver):
    def __init__(self, scip_params={}, additional_observers=[], expert_probability=1):
        observer = DistObserver(expert_probability=expert_probability)
        env = HistoryBranchingEnvironment(
            observation_function=(
                observer,
                *additional_observers,
            ),
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "time": ecole.reward.SolvingTime(),
            },
            scip_params=scip_params,
        )
        super().__init__(env)


    def __repr__(self):
        return f"dist"





class MixEcoleSolver(EcoleSolver):
    def __init__(self, solvers, scip_params):
        self.solvers, self.conditions = list(zip(*solvers))
        obsevers = []
        for solver in self.solvers:
            if isinstance(solver, EcoleSolver):
                obsevers.append(solver.env.observation_function)
            else:
                obsevers.append(None)
        observer = ConditionalObservers(obsevers, self.conditions)

        env = HistoryBranchingEnvironment(
            observation_function=(
                observer
            ),
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "time": ecole.reward.SolvingTime(),
            },
            scip_params=scip_params,
        )

        super().__init__(env)

    def get_action(self, action_set, observation) -> int:
        for solver, condition in zip(self.solvers, self.conditions):
            if condition(self.model):
                return solver.get_action(action_set, observation)

    def after_action(self):
        for solver, condition in zip(self.solvers, self.conditions):
            if condition(self.model):
                if not isinstance(solver, EcoleSolver):
                    old_value = self.model.getParam('branching/ecole::ReverseBranchrule/priority')
                    self.model.setParam('branching/ecole::ReverseBranchrule/priority', 0)
                    solver.configure()
                    self.model.optimize()
                    self.model.setParam('branching/ecole::ReverseBranchrule/priority', old_value)

                    return True

                else:
                    self.model.setParam('branching/ecole::ReverseBranchrule/priority', 536870911)
                    return False

    def before_start(self, path):
        for solver, condition in zip(self.solvers, self.conditions):
            if isinstance(solver, EcoleSolver):
                solver.before_start(path)

        return super().before_start(path)

    def __repr__(self):
        m = ",".join([str(s) for s in self.solvers])
        return f"Mix({m})"


class GnnSolver(EcoleSolver):
    def __init__(self, policy_path, scip_params={}, try_use_gpu=True):
        observer = GnnObserver(
            policy_path,
            ecole.observation.NodeBipartite(),
            try_use_gpu=try_use_gpu
        )
        env = HistoryBranchingEnvironment(
            observation_function=[observer],
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "time": ecole.reward.SolvingTime(),
            },
            scip_params=scip_params,
        )

        self.policy_path = policy_path

        super().__init__(env)

    def __str__(self):
        return "GNN({})".format(self.policy_path)


class ClassicSolver(Solver):
    def __init__(self, branching_policy, scip_params={}):
        super().__init__()
        self.model = pyscipopt.Model()
        self.branching_policy = branching_policy
        self.configure()
        self.model.setParams(scip_params)

    def configure(self):
        utils.configure(self.model, branching_policy=self.branching_policy)

    def solve(self, path: str):
        self.model.readProblem(path)
        self.model.optimize()

    def __str__(self):
        return self.branching_policy


if __name__ == "__main__":
    n_objects = 20
    bin_capacity = 100
    weight_bounds = (20, 22)
    instances = BinpackingGenerator(n_objects, bin_capacity, weight_bounds)
    instances.seed(0)

    solver = UBSolver(n_repetitions=20, scip_params={"limits/nodes": 2000}, expert_probability=0.1,
                      additional_observers=[ecole.observation.NodeBipartite()])

    # generator = DatasetGenerator(instances, solver)
    # generator.generate("test", 100)

    # solver = UBSolver(20, {}, None)
    solver = ClassicSolver("vanillafullstrong")
    instance = next(instances)
    m = instance.as_pyscipopt()
    m.writeProblem("prob.lp")
    solver.solve("prob.lp")

    print(solver["time"], solver["gap"], solver["obj"], solver["nnodes"])
