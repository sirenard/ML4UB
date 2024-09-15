import itertools
import math
import random
from collections import defaultdict

import numpy as np
import pyscipopt
from typing_extensions import override

from FixedObserver import FixedObserver
from solvers import EcoleSolver, ClassicSolver

class MCTS:
    def __init__(self, exploration_weight=1.4, heuristic_weight=1, max_iterations=1000):
        self.exploration_weight = exploration_weight
        self.heuristic_weight = heuristic_weight
        self.max_iterations = max_iterations

    def search(self, root):
        """
        Run MCTS from a given root node.
        """
        best_state = None
        best_score = 0

        for _ in range(self.max_iterations):
            node = self.select(root)
            if not node.is_terminal():
                node = self.expand(node)
            reward = node.simulate()
            if reward > best_score or best_state is None:
                best_state = node
                best_score = reward

            print(f"Best score: {best_score}, current: {reward}")
            node.backpropagate(reward)

        # Return the child with the highest visit count (most promising move)
        return best_state

    def select(self, node):
        """
        Select a node to explore using the UCB1 formula.
        """
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            else:
                node = node.best_child(self.exploration_weight, self.heuristic_weight)
        return node

    def expand(self, node):
        """
        Expand a node by trying one of its untried actions.
        """
        action = node.untried_actions.pop()
        child_node = node.apply_action(action)
        node.children.append(child_node)
        return child_node


# Example subclass for a specific problem (e.g., a game or optimization problem)
class RankingMCTSNode():
    def __init__(self, state, nnodes, instance_path, heuristic, parent=None):
        self.remaining = list(range(nnodes))
        self.nnodes = nnodes
        self.instance_path = instance_path
        self.heuristic = heuristic

        self.state = state  # The state of the problem (could be board state, partial solution, etc.)
        self.parent = parent  # Parent node in the MCTS tree
        self.children = []  # List of children nodes
        self.visits = 0  # Number of times this node has been visited
        self.reward = 0.0  # Cumulative reward from simulations
        self.untried_actions = self.get_legal_actions()  # Actions that haven't been tried yet


        for n in state:
            self.remaining.remove(n)

    def get_legal_actions(self):
        return self.remaining

    def apply_action(self, action):
        # Define how an action modifies the state
        new_state = self.state + [action]  # Example of a state update
        return RankingMCTSNode(new_state, self.nnodes, self.instance_path, self.heuristic, parent=self)

    def is_terminal(self):
        return len(self.remaining) == 0  # Example terminal condition

    def simulate(self):

        solver = EcoleSolver(FixedObserver({},
            score_function=lambda _: self.get_score()),
                             )

        solver.solve(self.instance_path)
        return -solver["nnodes"]  # Example of a random reward

    def get_score(self):
        score_dict = {f"t_x_{n}": self.nnodes - i for i, n in enumerate(self.state)}
        for n in self.remaining:
            score_dict[f"t_x_{n}"] = -1
        return score_dict

    def backpropagate(self, reward):
        """
        Backpropagate the simulation result to update node statistics.
        """
        self.visits += 1
        self.reward += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def is_fully_expanded(self):
        """
        Check if all possible actions from this state have been tried.
        """
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight=1.4, heuristic_weight=1):
        """
        Select the best child node using UCB1 (Upper Confidence Bound) formula.
        """
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            exploit = child.reward / child.visits
            explore = math.sqrt(2 * math.log(self.visits) / child.visits)
            h = self.heuristic(child.state[-1])
            ucb_score = exploit + exploration_weight * explore + heuristic_weight * h
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        return best_child
# Usage
def mcts_score(instance_path):
    solver = ClassicSolver("relpscost")
    solver.solve(instance_path)

    m: pyscipopt.Model = solver.model
    variables = m.getVars()

    sol = m.getBestSol()
    nnodes = m.getNNodes()

    dists = np.zeros(len(variables))
    for i in range(len(variables)):
        var = m.getTransformedVar(variables[i])
        ps_score = m.getVarPseudocostScore(var, sol[variables[i]])
        dists[i] = ps_score

    dists /= dists.max()
    def heuristic(j):

        if j==len(variables):
            return 1

        return dists[j]

    root = RankingMCTSNode([], len(variables), instance_path, heuristic=heuristic)
    mcts = MCTS(max_iterations=30, heuristic_weight=10000)
    best_node = mcts.search(root)

    return best_node.get_score()

def brut_force_score(instance_path):
    solver = ClassicSolver("relpscost")
    solver.solve(instance_path)

    m: pyscipopt.Model = solver.model
    variables = m.getVars()

    k = 10
    n = len(variables)

    sol = m.getBestSol()

    best_size = np.inf
    best_scores = None

    for comb in itertools.combinations(range(n), k):
        scores = {}
        for i,j in enumerate(comb):
            var = m.getTransformedVar(variables[j])
            scores[var.name] = len(variables)-i

        for var in variables:
            var = m.getTransformedVar(var)
            if var.name in scores:
                continue

            ps_score = m.getVarPseudocostScore(var, sol[var])
            scores[var.name] = -10**(8) + ps_score

        solver = EcoleSolver(FixedObserver({}, score_function=lambda _: scores))
        solver.solve(instance_path)

        s = solver["nnodes"]
        if s < best_size:
            best_size = s
            best_scores = scores

        print(f"Best: {best_size}, current: {s}")

    return best_scores

