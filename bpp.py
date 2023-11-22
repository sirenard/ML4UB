import ecole
import numpy as np
from pyscipopt import *


class BinpackingGenerator:
    def __init__(self, n_objects=15, bin_capacity=35, weight_bounds=(5, 20)):
        self.n = n_objects
        self.b = bin_capacity
        self.weight_bounds = weight_bounds
        self.rng = np.random.default_rng()

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self):
        model = self.build_instance()
        return ecole.scip.Model.from_pyscipopt(model)

    def build_instance(self):
        w = self.rng.integers(low=self.weight_bounds[0], high=self.weight_bounds[1], size=self.n)

        model = Model("bpp")

        x = {}
        for i in range(self.n):
            for j in range(i + 1):
                x[i, j] = model.addVar(vtype="B", name=f"x({i},{j})")

        for j in range(self.n):
            model.addCons(quicksum(w[i] * x[i, j] for i in range(j, self.n)) <= self.b * x[j, j], f"Capac({j})")

        for i in range(self.n):
            model.addCons(quicksum(x[i, j] for j in range(i + 1)) == 1, f"Assign({i})")

        model.setObjective(quicksum(x[j, j] for j in range(self.n)), "minimize")
        model.data = x

        return model
