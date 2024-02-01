import math
import random
import time

import ecole
import numpy as np
import pyscipopt

import reward
from environments import HistoryBranchingEnvironment
from mpi import MpiPool
from observer import ConditionalObservers, RandomObserver, Observer


class MctsObserver(Observer):
    def __init__(self, play_out_observer=ecole.observation.StrongBranchingScores(), simulation_time_budget=600, pool: MpiPool=MpiPool()):
        super().__init__()
        self.play_out_observer = play_out_observer
        self.simulation_time_budget = simulation_time_budget
        self.pool = pool

    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = m.getLPBranchCands()
        n_vars = int(m.getNVars())


        prob_indexes = sorted([var.getCol().getLPPos() for var in candidates])
        self.run_simulation(prob_indexes)

    def run_simulation(self, actions):
        root = MctsNode(None, None, actions)
        init_time = time.time()

        history_index = 0
        action_history = []

        env = HistoryBranchingEnvironment(
            observation_function=ConditionalObservers(
                    [RandomObserver(), self.play_out_observer],
                    [lambda m: history_index<len(action_history), lambda m: True]
                ),
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "gap": reward.gapReward()
            }
        )

        while init_time + self.simulation_time_budget > time.time():
            node = root.select(math.sqrt(2))
            action_history.clear()
            node.get_action_sequence(action_history)
            env.seed(0)

            observation, action_set, _, done, info = env.reset(self.instance_path)
            nb_nodes = info["nb_nodes"]

            history_index = 0

            expand_action = None
            print(action_history)
            while not done:
                if history_index < len(action_history):
                    action = action_history[history_index]
                    # assert action in action_set
                elif history_index == len(action_history):
                    action = random.choice(node.possible_actions)
                    expand_action = action

                    # assert action in action_set
                else:
                    if expand_action is not None:
                        node.expand(expand_action, action_set)
                        expand_action = None

                    action = action_set[observation[action_set].argmax()]

                history_index += 1

                # print(action, action_set)
                observation, action_set, _, done, info = env.step(action)
                nb_nodes += info["nb_nodes"]

            print(nb_nodes)

    def before_reset(self, model):
        pass

    def __str__(self):
        return "MCTS"








class MctsNode:
    def __init__(self, parent, action, possible_actions):
        self.total_reward = 0
        self.visits = 0
        self.parent = parent
        self.action = action

        self.possible_actions = possible_actions
        self.children = []

    def select(self, exploration_weight):
        if self.is_leaf():
            return self

        values = np.array([child.ucb_values(exploration_weight) for child in self.children])
        node = self.children[values.argmax()]
        return node.select(exploration_weight)

    def ucb_value(self, exploration_weight):
        if self.visits == 0:
            return float('inf')  # Handle unvisited nodes (division by zero)

        exploitation_term = self.total_reward / self.visits
        exploration_term = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

        ucb = exploitation_term + exploration_term
        return ucb

    def is_leaf(self):
        return len(self.possible_actions) > 0

    def reward(self, reward):
        self.total_reward += reward
        self.visits += 1

        if self.parent is not None:
            self.parent.reward(reward)

    def expand(self, action, new_actions):
        assert action in self.possible_actions

        child = MctsNode(self, action, new_actions)
        self.children.append(child)
        self.possible_actions.remove(action)

    def get_action_sequence(self, seq=None):
        if self.action is None:
            return seq

        if seq is None:
            seq = []
        if self.parent is not None:
            self.parent.get_action_sequence(seq)

        seq.append(self.action)
        return seq


