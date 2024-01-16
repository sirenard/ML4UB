import math
import tempfile

import torch

import utils
from model import load_policy, get_device
from mpi import MpiPool

import ecole
import numpy as np
from itertools import repeat

import pyscipopt

import reward
from environments import HistoryBranchingEnvironment


class UnrealisticObserver:
    def __init__(self, n_repetitions, expert_probability=1, seed=None, scip_params={}, local=False, pool: MpiPool = None):
        self.seed = 0
        self.instance_path = "prob.lp"
        self.action_history = []
        self.n_repetitions = n_repetitions
        self.expert_probability = expert_probability
        self.rng = np.random.default_rng(seed)
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.scip_params = scip_params
        self.pool: MpiPool = pool
        self.local = local

    def extract(self, model, done):
        self.extract_expert(model)

    def extract_expert(self, model):
        m:pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = m.getLPBranchCands()
        n_actions = len(candidates)
        n_vars = int(m.getNVars())

        prob_indexes = sorted([var.getCol().getLPPos() for var in candidates])

        instance_path = self.instance_path
        solution_path = None
        action_history = self.action_history

        if self.local:
            prob_file = tempfile.NamedTemporaryFile(suffix=".lp", dir="tmp")
            sol_file = tempfile.NamedTemporaryFile(suffix=".sol", dir="tmp")

            instance_path = prob_file.name
            solution_path = sol_file.name

            utils.write_to_file(instance_path, solution_path, m)
            action_history = []

        r = self.rng.integers(0, 2**32)
        args = list(zip(
            prob_indexes*self.n_repetitions,
            repeat(action_history),
            [r + j for _ in range(n_actions) for j in range(self.n_repetitions)],
            repeat(instance_path),
            repeat(self.scip_params),
            repeat(solution_path),
            repeat(self.n_repetitions == 1),
        ))

        if self.pool:
            res = self.pool.map(self.get_score, args)
        else:
            res = [self.get_score(*arg) for arg in args]

        if self.local:
            prob_file.close()
            sol_file.close()

        res = np.array(res).reshape(self.n_repetitions, n_actions)
        scores = res.min(axis=0)

        res = np.zeros(n_vars)
        res[:] = np.nan
        res[prob_indexes] = scores
        return res

    def before_reset(self, model):
        self.pseudocosts_function.before_reset(model)

    def reset(self, instance_path, seed):
        self.action_history = []
        self.instance_path = instance_path
        self.seed = seed

    def step(self, action):
        self.action_history.append(action)

    @staticmethod
    def get_score(action_index, action_history, seed, instance_path, scip_params, solution_path, use_full_strong=False):
        env = HistoryBranchingEnvironment(
            observation_function=ecole.observation.StrongBranchingScores() if use_full_strong else None,
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "gap": reward.gapReward()
            },
            scip_params=scip_params
        )


        rng = np.random.default_rng(seed)

        history_index = 0
        env.seed(0)

        nb_nodes, t = 0, 0

        observation, action_set, _, done, info = env.reset(instance_path, solution_path=solution_path)
        nb_nodes += info["nb_nodes"]

        while not done:
            if history_index < len(action_history):
                action = action_history[history_index]
                #assert action in action_set
            elif history_index == len(action_history):
                action = action_index
                #assert action in action_set
            elif use_full_strong:
                action = action_set[observation[action_set].argmax()]
            else:
                action = rng.choice(action_set)

            history_index += 1


            #print(action, action_set)
            observation, action_set, _, done, info = env.step(action)
            nb_nodes += info["nb_nodes"]

        gap = info["gap"]
        return (1+gap) * nb_nodes
        #return nb_nodes


class GnnObserver:
    def __init__(self, policy_path: str, feature_observer=ecole.observation.NodeBipartite()):
        self.policy = load_policy(policy_path)
        self.feature_observer = feature_observer

    def before_reset(self, model):
        self.feature_observer.before_reset(model)
    def extract(self, model, done):
        device = get_device()
        observation = self.feature_observer.extract(model, done)
        observation = (
            torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
            torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
            torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device),
            torch.from_numpy(observation.variable_features.astype(np.float32)).to(device),
        )

        logits = self.policy(*observation)
        return logits.detach().numpy()


class DistObserver:
    def __init__(self, expert_probability=1, seed=0):
        self.instance_path = None
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.sol=None
        self.rng = np.random.default_rng(seed)

    def extract(self, model, done):
        is_expert = self.rng.random() <= self.expert_probability
        if is_expert:
            scores = self.extract_expert(model)
        else:
            scores = self.pseudocosts_function.extract(model, done)
            scores = np.array(scores)

        return scores, is_expert

    def extract_expert(self, model):
        pyscipopt_model:pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = pyscipopt_model.getLPBranchCands()
        n_actions = len(candidates)
        n_vars = int(pyscipopt_model.getNVars())

        prob_indexes = sorted([var.getCol().getLPPos() for var in candidates])
        scores = []

        prob_file = tempfile.NamedTemporaryFile(suffix=".lp", dir="tmp")
        output = utils.OutputControl()
        output.mute()
        pyscipopt_model.writeProblem(prob_file.name, trans=True)
        output.unmute()


        for index, var in zip(prob_indexes, candidates):
            var: pyscipopt.scip.Variable
            lp_val = var.getLPSol()
            distances = []
            models = []
            for i in range(2):
                m = pyscipopt.Model()
                m.setParam("display/verblevel", 0)
                m.readProblem(prob_file.name)
                for v in m.getVars():
                    m.chgVarType(v, 'CONTINUOUS')
                    if v.name == var.name:
                        if i == 0:
                            m.tightenVarUbGlobal(v, math.floor(lp_val))
                        else:
                            m.tightenVarLbGlobal(v, math.ceil(lp_val))


                m.optimize()
                distances.append(self.get_dist(m))
                models.append(m)

            #dist=abs(distances[0]-distances[1])
            #dist = self.get_models_dist(*models) - min(distances)
            #dist = self.get_models_dist(*models)
            if np.inf in distances:
                distances.remove(np.inf)
                dist = distances[0]
            else:
                dist = abs(distances[0]-distances[1])
            scores.append(dist)


        prob_file.close()

        res = np.zeros(n_vars)
        res[:] = np.nan
        res[prob_indexes] = scores
        return res

    def before_reset(self, model):
        self.pseudocosts_function.before_reset(model)

        m = pyscipopt.Model()
        m.setParam("display/verblevel", 0)
        m.readProblem(self.instance_path)
        m.optimize()
        best_sol = m.getBestSol()

        self.sol = {}
        for var in m.getVars():
            var: pyscipopt.scip.Variable
            t_var = m.getTransformedVar(var)

            val = best_sol[var]

            self.sol[t_var.name] = val


    def reset(self, instance_path, seed):
        self.action_history = []
        self.instance_path = instance_path
        self.seed = seed

    def step(self, action):
        self.action_history.append(action)

    def get_dist(self, model: pyscipopt.Model):
        if model.getStatus() != "optimal":
            return np.inf

        sol = model.getBestSol()
        dist = 0
        for var in model.getVars():
            t_var = model.getTransformedVar(var)
            if var.name in self.sol:
                dist += (sol[t_var] - self.sol[var.name])**2

        return dist**0.5

    def get_models_dist(self, m1: pyscipopt.Model, m2: pyscipopt.Model):
        if m1.getStatus() != "optimal" or m2.getStatus() != "optimal":
            return np.inf

        vars = {}

        for m in (m1, m2):
            sol = m.getBestSol()
            for var in m.getVars():
                name = var.name
                if name not in vars:
                    vars[name] = []

                vars[name].append(sol[var])

        dist = 0
        for var in vars:
            dist += (vars[var][0] - vars[var][1])**2

        return dist**2


class ConditionalObservers:
    def __init__(self, observers, conditions):
        self.observers = observers
        self.conditions = conditions
        self.last_observer_index_used = -1

    def before_reset(self, model):
        for observer in self.observers:
            if observer is not None:
                observer.before_reset(model)

    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        for i, (observer, condition) in enumerate(zip(self.observers, self.conditions)):
            if condition(m):
                self.last_observer_index_used = i
                if observer is None:
                    return
                else:
                    return observer.extract(model, done)

    def get_last_observer_index_used(self):
        return self.last_observer_index_used
