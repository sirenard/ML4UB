import ecole
import numpy as np

from environments import HistoryBranchingEnvironment


class UnrealisticObserver:
    def __init__(self, n_repetitions):
        self.seed = 0
        self.instance_path = None
        self.action_history = []
        self.n_repetitions = n_repetitions

    def extract(self, model, done):
        n_actions = len(model.as_pyscipopt().getLPBranchCands()[0])

        #scores = np.array([self.get_score(i, self.action_history[:], self.seed, self.instance_path, self.n_repetitions) for i in range(n_actions)])
        scores = []
        for i in range(n_actions):
            score = self.get_score(i, self.action_history[:], self.seed, self.instance_path, self.n_repetitions)
            scores.append(score)
        scores = np.array(scores)

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
    def get_score(action_index, action_history, seed, instance_path, n_repetitions):
        env = HistoryBranchingEnvironment(
            information_function={
                "nb_nodes": ecole.reward.NNodes()
            }
        )

        scores = np.zeros(n_repetitions)

        rng = np.random.default_rng(seed)

        for i in range(n_repetitions):
            history_index = 0
            env.seed(seed)

            nb_nodes, t = 0, 0
            #instance = ecole.scip.Model.from_file(instance_path)
            instance = instance_path


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

            scores[i] = nb_nodes

        return scores.min()


