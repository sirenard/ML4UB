import os
import pickle

import ecole
import numpy as np
import random

import pyscipopt

from FixedObserver import FixedObserver
from environments import HistoryBranchingEnvironment
from observer import GnnObserver, StrongBranching
from solvers import EcoleSolver, ClassicSolver


class AntColonySystem:
    def __init__(self, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit, problem_size, n_proc=24):
        self.num_ants = num_ants  # Number of ants
        self.num_iterations = num_iterations  # Number of iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta  # Heuristic importance
        self.evaporation_rate = evaporation_rate  # Pheromone evaporation rate
        self.pheromone_deposit = pheromone_deposit  # Amount of pheromone deposited
        self.problem_size = problem_size  # Number of nodes/vertices
        self.pheromone = np.ones((problem_size, problem_size))  # Initial pheromone matrix
        self.heuristic = np.ones((problem_size, problem_size))  # Heuristic matrix (e.g., 1/distance)
        self.best_solution = None
        self.best_cost = float('inf')
        self.n_proc = n_proc

    def initialize_heuristic(self, heuristic_function):
        """ Initialize the heuristic matrix using a heuristic function (e.g., 1/distance) """
        for i in range(self.problem_size):
            for j in range(self.problem_size):
                if i != j:
                    self.heuristic[i][j] = heuristic_function(i, j)
                else:
                    self.heuristic[i][j] = 0

    def choose_next_node(self, current_node, available_nodes):
        """ Choose the next node based on pheromone and heuristic values """
        probabilities = []
        for node in available_nodes:
            probability = (self.pheromone[current_node][node] ** self.alpha) * (
                        self.heuristic[current_node][node] ** self.beta)
            probabilities.append(probability)

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()  # Normalize probabilities
        return np.random.choice(available_nodes, p=probabilities)

    def update_pheromone(self, ants_solutions):
        """ Update the pheromone levels on the paths used by the ants """
        # Evaporate pheromone
        self.pheromone *= (1 - self.evaporation_rate)

        # Deposit new pheromones
        for solution, cost in ants_solutions:
            pheromone_to_add = self.pheromone_deposit / cost
            for i in range(len(solution) - 1):
                from_node = solution[i]
                to_node = solution[i + 1]
                self.pheromone[from_node][to_node] += pheromone_to_add
                self.pheromone[to_node][from_node] += pheromone_to_add  # Assuming undirected problem

    def solve(self, cost_function):
        """ Main ACS loop to solve the problem """
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
                    solution = []
                    #current_node = random.randint(0, self.problem_size - 1)  # Start at a random node
                    current_node = self.problem_size - 1
                    solution.append(current_node)

                    available_nodes = list(range(self.problem_size))
                    available_nodes.remove(current_node)

                    # Build the solution for the ant
                    while available_nodes:
                        next_node = self.choose_next_node(current_node, available_nodes)
                        solution.append(next_node)
                        available_nodes.remove(next_node)
                        current_node = next_node

                    # Shift the solution
                    while solution[0] != self.problem_size - 1:
                        s = solution.pop(0)
                        solution.append(s)
                    cost = cost_function(solution)

                    #ants_solutions.append((solution, cost))

                    w = os.fdopen(w, 'wb')
                    w.write(pickle.dumps((solution, cost)))
                    w.close()
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
                if cost < round_best:
                    round_best = cost
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = solution

            self.update_pheromone(ants_solutions)
            print(f"{iteration+1}/{self.num_iterations}: {self.best_cost}, round: {round_best}")

        return self.best_solution, self.best_cost


def acs_score(instance_path):
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

    def distance(i, j):

        if j==len(variables):
            return 1

        return dists[j]

        var1 = m.getTransformedVar(variables[i])
        var2 = m.getTransformedVar(variables[j])

        s1 = m.getVarPseudocostScore(var1, sol[variables[i]])
        s2 = m.getVarPseudocostScore(var2, sol[variables[j]])
        dist = abs(s1 - s2)

        return dist if dist!=0 else 1

    scip_params = {
        "estimation/method": "c",
        "estimation/completiontype": "m",
        "nodeselection/dfs/stdpriority": 1073741823,
        # "limits/nodes":1000,
        "limits/time": 300,
    }

    def cost_function(solution):
        """Example cost function (e.g., sum of distances for TSP)"""
        solver = EcoleSolver(FixedObserver(scip_params,
                                           score_function= lambda path: {m.getTransformedVar(variables[j]).name:-i for i,j in zip(range(len(variables)),solution[1:])} ),
                             scip_params
                             )
        solver.solve(instance_path)
        return solver["nnodes"]


    # Parameters
    num_ants = 10
    num_iterations = 30
    alpha = 4  # Pheromone importance
    beta = 2  # Heuristic importance
    evaporation_rate = 0.5
    pheromone_deposit = nnodes

    problem_size = len(variables) + 1
    # Create the Ant Colony System
    acs = AntColonySystem(num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit, problem_size)

    # Initialize heuristic (with a distance function as heuristic)
    acs.initialize_heuristic(distance)

    # Solve the problem (with a simple cost function)
    best_solution, best_cost = acs.solve(cost_function)

    scores = {}
    for i,j in zip(range(len(variables)),best_solution[1:]):
        var = m.getTransformedVar(variables[j])
        scores[var.name] = -i

    return scores


def sb_score(instance_path):
    observer = StrongBranching()

    solver = EcoleSolver(StrongBranching())
    solver.solve(instance_path)

    m = solver.model

    history = solver.env.history

    scores = {}
    for i, var in enumerate(m.getVars()):
        #prob_index = var.getCol().getLPPos()

        if i in history:
            pos = len(history) - history.index(i)
        else:
            pos = -1
        scores[f"t_x_{i}"] = pos

    return scores

