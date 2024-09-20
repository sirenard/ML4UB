import os
import pickle
import tempfile
from itertools import product

import ecole
import numpy as np
import random

import pyscipopt
from dill.pointers import children
from sympy.series.limits import heuristics

import utils
from FixedObserver import FixedObserver
from environments import HistoryBranchingEnvironment
from observer import GnnObserver, StrongBranching, Observer
from solvers import EcoleSolver, ClassicSolver

class ACSObserer(Observer):
    def __init__(self, **kwargs):
        super().__init__()
        self.acs = AntColonySystem(**kwargs)
        self.vars = []

    def reset(self, instance_path, seed=None):
        self.vars, _ = self.acs.solve(instance_path)
        #node = self.acs.root


    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = m.getLPBranchCands()
        n_vars = int(m.getNVars())

        prob_indexes = sorted([var.getCol().getLPPos() for var in candidates])

        scores = np.zeros(n_vars)
        scores[:] = np.nan
        scores[prob_indexes] = 0


        v = self.vars.pop(0)
        scores[v] = 1

        return scores

    def __str__(self):
        return "ACS"

class Node():
    def __init__(self):
        self.pheromone = 0
        self.children = {}

    def evaporate(self, evaporation_rate):
        self.pheromone *= 1-evaporation_rate
        for index in self.children:
            self.children[index].evaporate(evaporation_rate)

class AntObserver(Observer):
    def __init__(self, root: Node, alpha, beta):
        super().__init__()
        self.sb = ecole.observation.StrongBranchingScores()
        self.node: Node = root
        self.actions = []
        self.alpha = alpha
        self.beta = beta

    def before_reset(self, model):
        self.sb.before_reset(model)
        self.actions = []

    def extract(self, model, done):
        sb_scores = self.sb.extract(model, done)

        m: pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = m.getLPBranchCands()
        n_vars = int(m.getNVars())

        prob_indexes = sorted([var.getCol().getLPPos() for var in candidates])

        heuristics = np.zeros(len(prob_indexes))
        pheromones = np.zeros(len(prob_indexes))
        for i,index in enumerate(prob_indexes):
            heuristic = sb_scores[index]
            if index not in self.node.children:
                self.node.children[index] = Node()
            pheromone = self.node.children[index].pheromone

            heuristics[i] = heuristic
            pheromones[i] = pheromone


        # Choose the candidate
        probabilities = pheromones ** self.alpha * heuristics ** self.beta

        if probabilities.sum() == 0:
            probabilities = heuristics ** self.beta

        probabilities /= probabilities.sum()  # Normalize probabilities
        choice =  np.random.choice(prob_indexes, p=probabilities)

        scores = np.zeros(n_vars)
        scores[:] = np.nan
        scores[prob_indexes] = 0
        scores[choice] = 1

        self.node = self.node.children[choice]
        self.actions.append(choice)
        return scores


class AntColonySystem:

    def __init__(self, num_ants=10, num_iterations=30, alpha=2., beta=4., evaporation_rate=0.5, pheromone_deposit=1000, n_proc=24):
        self.num_ants = num_ants  # Number of ants
        self.num_iterations = num_iterations  # Number of iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta  # Heuristic importance
        self.evaporation_rate = evaporation_rate  # Pheromone evaporation rate
        self.pheromone_deposit = pheromone_deposit  # Amount of pheromone deposited
        self.best_solution = None
        self.best_cost = float('inf')
        self.n_proc = n_proc
        self.root = Node()


    def update_pheromone(self, ants_solutions):
        """ Update the pheromone levels on the paths used by the ants """
        # Evaporate pheromone
        self.root.evaporate(self.evaporation_rate)

        # Deposit new pheromones
        for solution, cost in ants_solutions:
            pheromone_to_add = self.pheromone_deposit / cost
            node = self.root
            for index in solution:
                node.children[index].pheromone += pheromone_to_add
                node = node.children[index]

    def solve(self, path: str):
        """ Main ACS loop to solve the problem """
        self.root = Node()
        self.best_solution = None
        self.best_cost = float('inf')
        for iteration in range(self.num_iterations):
            ants_solutions = []

            pipes = [os.pipe() for _ in range(self.num_ants)]
            pids = []
            seed = random.randint(1, 9999999)
            for ant in range(self.num_ants):
                pid = os.fork()
                r, w = pipes[ant]
                if pid > 0:  # parent
                    pids.append(pid)
                    os.close(w)
                    if len(pids) == self.n_proc:
                        p = pids[0]
                        os.waitpid(p, 0)
                        pids.remove(p)
                else: # child
                    os.close(r)
                    np.random.seed(seed + ant)
                    solution, cost = self.build_solution(path)

                    #ants_solutions.append((solution, cost))
                    w = os.fdopen(w, 'wb')
                    w.write(pickle.dumps((solution, cost)))
                    w.close()
                    #print(f"Ant {ant}: {cost}")
                    os._exit(os.EX_OK)

            for pid in pids:
                os.waitpid(pid, 0)

            round_best = np.inf
            for ant in range(self.num_ants):
                r, _ = pipes[ant]

                r = os.fdopen(r, "rb")
                str = r.read()
                (solution, cost) = pickle.loads(str)
                r.close()

                node = self.root
                ants_solutions.append((solution, cost))
                for index in solution:
                    if index not in node.children:
                        node.children[index] = Node()

                    node = node.children[index]

                if cost < round_best:
                    round_best = cost
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = solution

            self.update_pheromone(ants_solutions)
            #print(f"{iteration+1}/{self.num_iterations}: {self.best_cost}, round: {round_best}")

        return self.best_solution, self.best_cost

    def build_solution(self, path):
        observer = AntObserver(self.root, self.alpha, self.beta)
        solver = EcoleSolver(observer, {})
        solver.solve(path)
        return observer.actions, solver["nnodes"]



def acs(instance_path, num_ants = 10, num_iterations = 30, alpha = 4., beta = 2., evaporation_rate = 0.5, pheromone_deposit = 1000):

    acs = AntColonySystem(num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit)

    best_solution, best_cost = acs.solve(instance_path)

    return best_cost


if __name__ == "__main__":
    instances = ecole.instance.CombinatorialAuctionGenerator( n_items=100, n_bids=500)
    instances.seed(12)
    #next(instances)


    output_control = utils.OutputControl()
    prob_file = tempfile.NamedTemporaryFile(suffix=".lp", dir="tmp")

    solver = EcoleSolver(StrongBranching())

    # instance = next(instances)
    # output_control.mute()
    # instance.as_pyscipopt().writeProblem(prob_file.name)
    # output_control.unmute()
    #
    # solver.solve(prob_file.name)
    # print(f"SB: {solver["nnodes"]}")
    #
    # alpha_values = np.linspace(0.1, 2.0, 5)
    # beta_values = np.linspace(1, 5, 5)
    # rho_values = np.linspace(0.1, 0.9, 3)
    #
    # best_params = None
    # best_performance = float('inf')
    #
    # for alpha, beta, rho in product(alpha_values, beta_values, rho_values):
    #
    #     performance = acs(prob_file.name,
    #         alpha=alpha,
    #         beta=beta,
    #         evaporation_rate=rho,
    #         pheromone_deposit=1000,
    #         num_ants=24,
    #         )
    #
    #     if performance < best_performance:
    #         best_performance = performance
    #         best_params = (alpha, beta, rho)
    #
    #     print(f"{alpha},{beta},{rho}: {performance}. Best so far: {best_performance}")
    #
    #
    # print(f"Best: {best_params}, {best_performance}")



    for i, instance in zip(range(10), instances):
        output_control.mute()
        instance.as_pyscipopt().writeProblem(prob_file.name)
        output_control.unmute()

        solver.solve(prob_file.name)
        print(f"SB: {solver["nnodes"]}")

        score = acs(prob_file.name,
            alpha=1.525,
            beta=5,
            evaporation_rate=0.1,
            )
        print(score)


