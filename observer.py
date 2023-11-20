from multiprocessing import Pool

import ecole
import numpy as np
from itertools import repeat

from environments import HistoryBranchingEnvironment


class UnrealisticObserver:
    def __init__(self, n_repetitions, n_cpus):
        self.seed = 0
        self.instance_path = None
        self.action_history = []
        self.n_repetitions = n_repetitions
        self.n_cpus = n_cpus

    def extract(self, model, done):
        n_actions = len(model.as_pyscipopt().getLPBranchCands()[0])

        args = zip(
            list(range(n_actions))*self.n_repetitions,
            repeat(self.action_history),
            [self.seed + j for _ in range(n_actions) for j in range(self.n_repetitions)],
            repeat(self.instance_path),
        )

        with Pool(self.n_cpus) as pool:
            res = pool.starmap(self.get_score, args)
        #res = [self.get_score(*arg) for arg in args]

        res = np.array(res).reshape(self.n_repetitions, n_actions)
        scores = res.min(axis=0)


        #scores = np.array([self.get_score(i, self.action_history[:], self.seed, self.instance_path, self.n_repetitions) for i in range(n_actions)])
        # scores = []
        # for i in range(n_actions):
        #     score = self.get_score(i, self.action_history[:], self.seed, self.instance_path, self.n_repetitions)
        #     scores.append(score)
        # scores = np.array(scores)

        return scores

    def before_reset(self, model):
        pass
    def reset(self, instance_path, seed):
        self.action_history = []
        self.instance_path = instance_path
        self.seed = seed

    def step(self, action):
        self.action_history.append(action)

    @staticmethod
    def get_score(action_index, action_history, seed, instance_path):
        env = HistoryBranchingEnvironment(
            information_function={
                "nb_nodes": ecole.reward.NNodes()
            }
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

        return nb_nodes


