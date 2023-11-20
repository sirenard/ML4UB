import ecole
import numpy as np
from pyscipopt import *


def FFD(s, B):
    remain = [B]
    sol = [[]]
    for item in sorted(s, reverse=True):
        for j,free in enumerate(remain):
            if free >= item:
                remain[j] -= item
                sol[j].append(item)
                break
        else:
            sol.append([item])
            remain.append(B-item)
    return sol


def bpp(s,B):
    n = len(s)
    U = len(FFD(s,B))
    model = Model("bpp")
    x,y = {},{}
    for i in range(n):
        for j in range(U):
            x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))
    for j in range(U):
        y[j] = model.addVar(vtype="B", name="y(%s)"%j)
    for i in range(n):
        model.addCons(quicksum(x[i,j] for j in range(U)) == 1, "Assign(%s)"%i)
    for j in range(U):
        model.addCons(quicksum(s[i]*x[i,j] for i in range(n)) <= B*y[j], "Capac(%s)"%j)
    for j in range(U):
        for i in range(n):
            model.addCons(x[i,j] <= y[j], "Strong(%s,%s)"%(i,j))
    model.setObjective(quicksum(y[j] for j in range(U)), "minimize")
    model.data = x,y
    return model


def binpackingGenerator(seed=0, n_objects=15, bin_capacity=35, weight_bounds=(5,20)):
    rng = np.random.default_rng(seed)
    while True:
        #w = random_weights(rng, n_objects, weight_bounds[0], weight_bounds[1])
        w = rng.integers(low=weight_bounds[0], high=weight_bounds[1], size=n_objects)
        instance = bpp(w, bin_capacity)
        yield ecole.scip.Model.from_pyscipopt(instance)
