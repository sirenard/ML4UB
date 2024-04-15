import time

import ecole
import numpy as np
import pyscipopt

import utils
from environments import HistoryBranchingEnvironment
from observer import Observer


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

    def __init__(self, score_observer: Observer, scip_params={}, additional_observers=[], before_start_callbacks=[],
                 before_action_callbacks=[]):
        super().__init__()
        self.observer = score_observer
        self.env: ecole.environment = HistoryBranchingEnvironment(
            observation_function=(
                score_observer,
                *additional_observers
            ),
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "time": ecole.reward.SolvingTime(),
            },
            scip_params=scip_params,
        )

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
        self.observer.reset(path)
        self.before_start(path)
        self.env.seed(0)

        t = time.time()
        observation, action_set, _, done, info = self.env.reset(instance)
        self.model = instance.as_pyscipopt()

        nb_nodes = info["nb_nodes"]
        while not done:
            action = self.get_action(action_set, observation)
            self.observer.step(action)
            self.before_action(action, action_set, observation)
            print(f"Action {action}")

            observation, action_set, _, done, info = self.env.step(action)
            print("DOne")
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
        #action_index = scores[action_set].argmax()
        action_index = np.random.choice(np.flatnonzero(scores[action_set] == scores[action_set].max())) # chose at random an index with max value
        action = action_set[action_index]
        return action

    def __str__(self):
        return str(self.observer)


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
