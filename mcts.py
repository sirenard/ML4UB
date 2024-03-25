import math
import random
import tempfile
import time

import ecole
import numpy as np
import pyscipopt

import reward
import utils
from environments import HistoryBranchingEnvironment
from mpi import MpiPool
from observer import ConditionalObservers, RandomObserver, Observer
from uc import UcigGenerator


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 1
        self.score = 0.0

    def expand(self):
        possible_states = self.state.get_possible_states()
        self.env = None
        for state in possible_states:
            self.children.append(Node(state, self))

    def select_child(self, exploration_constant=1.4):
        selected_child = None
        max_ucb = -float('inf')
        for child in self.children:
            ucb = (child.score / child.visits) + exploration_constant * math.sqrt(math.log(self.visits) / child.visits)

            if ucb > max_ucb:
                selected_child = []
                max_ucb = ucb

            if ucb == max_ucb:
                selected_child.append(child)

        return random.choice(selected_child)

    def update(self, score):
        self.visits += 1
        self.score += score

class MonteCarloTreeSearch:
    def __init__(self, state, simulation_limit=float('inf')):
        self.root = Node(state)
        self.simulation_limit = simulation_limit

    def best_action(self, simulations_count=200):
        for i in range(simulations_count):
            print(f"{i+1}/{simulations_count}")
            node = self.select_node()
            score = self.simulate(node)
            self.backpropagate(node, score)
        return self.get_best_child().state.get_last_action()

    def select_node(self):
        current_node = self.root
        while current_node.children:
            current_node = current_node.select_child()
        current_node.expand()
        return current_node

    def simulate(self, node):
        state = node.state.clone()
        i = 0
        while not state.is_terminal() and i<self.simulation_limit:
            action = random.choice(state.get_possible_actions())
            state.apply_action(action)
            i += 1
        return state.get_score()

    def backpropagate(self, node, score):
        while node:
            node.update(score)
            node = node.parent

    def get_best_child(self):
        return max(self.root.children, key=lambda child: child.visits)

# Example Usage
class GameState:
    def __init__(self, history, instance):
        self.history = history[:]
        self.instance = instance
        self.is_done = False

        self.env = HistoryBranchingEnvironment(
            observation_function=RandomObserver(),
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                },
            scip_params={"estimation/method": 'c'}
        )
        self.env.seed(0)

        observation, action_set, _, self.is_done, info = self.env.reset(instance)
        self.nb_nodes = info["nb_nodes"]
        for action in history:
            observation, action_set, _, self.is_done, info = self.env.step(action)
            self.nb_nodes += info["nb_nodes"]


        self.possible_actions = action_set

    def clone(self):
        return GameState(self.history, self.instance)

    def is_terminal(self):
        # Implement terminal state check logic here
        return self.is_done

    def get_possible_actions(self):
        # Return list of possible actions
        return self.possible_actions

    def apply_action(self, action):
        # Apply action to update state
        observation, action_set, _, self.is_done, info = self.env.step(action)
        self.nb_nodes += info["nb_nodes"]
        self.possible_actions = action_set

    def get_possible_states(self):
        # Return list of possible successor states
        res = [self.clone() for action in self.get_possible_actions()]
        for i, action in enumerate(self.get_possible_actions()):
            res[i].apply_action(action)

        return res


    def get_last_action(self):
        return self.history[-1]

    def get_score(self):
        # Return score for the current state
        model: pyscipopt.Model = self.env.model.as_pyscipopt()
        estimation = model.getTreesizeEstimation()
        nnodes = model.getNNodes()
        return 1/math.log2(estimation)


class MctsObserver(Observer):
    def __init__(self, estimation_method='w', simulation_count=100, limits_rollout=6 ):
        super().__init__()
        self.estimation_method = estimation_method
        self.limits_rollout = limits_rollout
        self.simulation_count = simulation_count
        self.file = None

    def before_reset(self, model):
        if self.file is not None:
            self.file.close()

        output_control = utils.OutputControl()
        self.file = tempfile.NamedTemporaryFile(suffix=".lp", dir="tmp")
        output_control.mute()
        model.as_pyscipopt().writeProblem(self.file.name)
        output_control.unmute()

    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        state = GameState(history=self.action_history, instance=self.file.name)
        mcts = MonteCarloTreeSearch(state, simulation_limit=self.limits_rollout)

        action = mcts.best_action(self.simulation_count)


        return action

    def __str__(self):
        return "Mcts"



if __name__ == "__main__":
    instances = UcigGenerator("instances", start_at_instance=5000, days=1)

    instance = next(instances)

    output_control = utils.OutputControl()
    instance_file = tempfile.NamedTemporaryFile(suffix=".lp", dir="tmp")
    instance.as_pyscipopt().writeProblem(instance_file.name)

    state = GameState(history=[], instance=instance_file.name)
    mcts = MonteCarloTreeSearch(state, simulation_limit=6)

    best_action = mcts.best_action()

    instance_file.close()


