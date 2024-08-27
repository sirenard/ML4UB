import pickle

import numpy as np
from scipy.stats import ttest_ind

from evaluation_tools import Data


def sgmean(xs: np.array, s: float) -> float:
    k = xs.size
    xs = xs + s
    xs[np.where(xs < 1)] = 1
    mean = np.prod(xs**(1/k))
    return mean-s


if __name__ == '__main__':
    data: Data = pickle.load(open("results/gpu/dataHard.pkl", "rb"))
    #data.remove_solver("GNN(uc_5_1_ub_1_0.6_local.pkl)")
    #data.remove_solver("GNN(uc_5_1_ub_2_0.6_local.pkl)")
    #data.remove_solver("GNN(uc_5_1_ub_4_0.6_local.pkl)")

    #, data = data.split_instances_over("gap", lambda g: g==0)

    metrics = data.metrics

    print(f"{'name':<40}" + "".join([f"{n:<11}{f"{'wins':<11}" if n == "time" else ""}" for n in metrics]))

    for i, name in enumerate(data.solvers):
        res = []


        indexes_solved = np.where(data.data[:, i, data.metrics.index("gap")] == 0)[0]
        indexes_unsolved = np.where(data.data[:, i, data.metrics.index("gap")] > 0)[0]
        nsolved = len(indexes_solved)

        for metric in metrics:
            metric_index = data.metrics.index(metric)
            s = {"nnodes": 100, "time": 1, "gap": 1}[metric]
            xs = data.data[:,i,metric_index]

            if metric == "nnodes":
                # remove unsolved instances
                xs = xs[indexes_solved]
            elif metric == "gap":
                xs = xs[indexes_unsolved]

            val = sgmean(xs, s) if xs.size else 0
            if metric == "gap":
                res.append(f"{val:<.3f}")
            else:
                res.append(f"{val:<.1f}")

            if metric == "time":
                # number of wins w.r.t time over number of solved instances
                n_wins = np.sum(np.isclose(data.data[indexes_solved,:,metric_index].min(axis=1), xs[indexes_solved]))
                res.append(f"{n_wins}/{indexes_solved.size}")

        print(f"{name:<40}& " + "& ".join([f"{x:<9}" for x in res]))

    res = data.performance_profile("time")
    print(res)
    res = data.performance_profile("nnodes")
    print(res)

    sb = data.data[:,data.solvers.index("SB"),data.metrics.index("nnodes")]
    rb = data.data[:,data.solvers.index("relpscost"),data.metrics.index("nnodes")]
    lsb = data.data[:,data.solvers.index("GNN(uc_5_1_sb_0.6.pkl)"),data.metrics.index("nnodes")]
    llb = data.data[:,data.solvers.index("GNN(uc_5_1_ub_3_0.6_local.pkl)"),data.metrics.index("nnodes")]
    print(ttest_ind(rb, llb))
    print(ttest_ind(rb, lsb))
    print(ttest_ind(llb, lsb))

