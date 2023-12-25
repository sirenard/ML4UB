from mpi import MpiPool

import ecole
import numpy as np
from itertools import repeat

import pyscipopt

import reward
from environments import HistoryBranchingEnvironment


class UnrealisticObserver:
    def __init__(self, n_repetitions, expert_probability=1, seed=None, scip_params={}, pool: MpiPool = None):
        self.seed = 0
        self.instance_path = None
        self.action_history = []
        self.n_repetitions = n_repetitions
        self.expert_probability = expert_probability
        self.rng = np.random.default_rng(seed)
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.scip_params = scip_params
        self.pool: MpiPool = pool

    def extract(self, model, done):
        is_expert = self.rng.random() <= self.expert_probability
        if is_expert:
            scores = - self.extract_expert(model)
        else:
            scores = self.pseudocosts_function.extract(model, done)
            scores = np.array(scores)

        return scores, is_expert

    def extract_expert(self, model):
        m:pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = m.getLPBranchCands()
        n_actions = len(candidates)
        n_vars = int(m.getNVars())

        prob_indexes = sorted([var.getCol().getLPPos() for var in candidates])

        args = list(zip(
            list(range(n_actions))*self.n_repetitions,
            repeat(self.action_history),
            [self.seed + j for _ in range(n_actions) for j in range(self.n_repetitions)],
            repeat(self.instance_path),
            repeat(self.scip_params)
        ))

        if self.pool:
            res = self.pool.map(self.get_score, args)
        else:
            res = [self.get_score(*arg) for arg in args]

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
    def get_score(action_index, action_history, seed, instance_path, scip_params):
        env = HistoryBranchingEnvironment(
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "gap": reward.gapReward()
            },
            scip_params=scip_params
        )

        rng = np.random.default_rng(seed)

        history_index = 0
        env.seed(seed)

        nb_nodes, t = 0, 0
        instance = ecole.scip.Model.from_file(instance_path)


        observation, action_set, _, done, info = env.reset(instance)
        nb_nodes += info["nb_nodes"]

        while not done:
            if history_index < len(action_history):
                action = action_history[history_index]
                assert action in action_set
            elif history_index == len(action_history):
                action = action_set[action_index]
            else:
                action = rng.choice(action_set)

            history_index += 1


            #print(action, action_set)
            observation, action_set, _, done, info = env.step(action)
            nb_nodes += info["nb_nodes"]

        gap = info["gap"]
        return (1+gap) * nb_nodes
        #return nb_nodes




class ExploreThenStrongBranch:
    """
    This custom observation function class will randomly return either strong branching scores (expensive expert)
    or pseudocost scores (weak expert for exploration) when called at every node.
    """

    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        """
        This function will be called at initialization of the environment (before dynamics are reset).
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        """
        Should we return strong branching or pseudocost scores at time node?
        """
        probabilities = [1 - self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)