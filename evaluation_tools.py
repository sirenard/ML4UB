import tempfile

import numpy as np

import utils
from solvers import Solver

class Data:
    def __init__(self, raw_data: np.array, solvers, metrics):
        self.data = raw_data
        self.solvers = solvers
        self.metrics = metrics

    def split_instances_over(self, metric: str, condition):
        assert metric in self.metrics, "Cannot make a split on a non-existing metric"

        index = self.metrics.index(metric)
        d=self.data[:,:,index] # keep only the compared metrix

        indexes = np.where(np.apply_along_axis(condition, 1, d))[0]

        positives = self.data[indexes,]
        negatives = np.delete(self.data, indexes, axis=0)
        return Data(positives, self.solvers, self.metrics), Data(negatives, self.solvers, self.metrics)

    def sum_over_instances(self):
        return self.data.sum(axis=0)

def evaluate_solvers(solvers: [Solver], instances, n_instances, metrics):
    print(f"{'Instance':<15}" + "".join([f"{str(solver):<{15*len(metrics)}}" for solver in solvers]))
    print(f"{'':<15}" + len(solvers)*("".join([f"{metric:<15}" for metric in metrics])))
    output_control = utils.OutputControl()

    data = np.zeros((n_instances, len(solvers), len(metrics)))

    prob_file = tempfile.NamedTemporaryFile(suffix=".lp", dir="tmp")

    crashed_instance_indexes = []

    for i, instance in zip(range(n_instances), instances):
        output_control.mute()
        instance.as_pyscipopt().writeProblem(prob_file.name)
        output_control.unmute()

        print(f"{i:<15}", end="")
        try:
            for j, solver in enumerate(solvers):
                solver.solve(prob_file.name)
                line = [solver[metric] for metric in metrics]
                for k, d in enumerate(line):
                    data[i, j, k] = d
                print("".join([f"{d:{'<15.3f' if type(d)==float else '<15'}}" for d in line]), end="", flush=True)
        except:
            crashed_instance_indexes.append(i)
            print("CRASH", end="")

        data = np.delete(data, crashed_instance_indexes, axis=0)

        print()

    res = Data(data, solvers, metrics)

    print("="*(15*(len(solvers)*len(metrics)+1)))

    for d, name in zip(res.split_instances_over("gap", lambda xs: xs.sum()==0), ["gap=0", "gap>0"]):
        sum = d.sum_over_instances()
        print(f"{'sum ' + name:<15}"+"".join([f"{sum[j,k]:<15.3f}" for j in range(len(solvers)) for k in range(len(metrics))]))

    prob_file.close()
    return res

