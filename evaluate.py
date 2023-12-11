from solvers import *
import numpy as np

def main(pool: MPIPool = None):
    n_objects = 20
    bin_capacity = 100
    weight_bounds = (20, 22)
    #instances = binpackingGenerator(0, n_objects, bin_capacity, weight_bounds)
    instances = BinpackingGenerator(n_objects, bin_capacity, weight_bounds)
    instances.seed(0)

    #instances = ecole.instance.SetCoverGenerator(n_rows=200, n_cols=400, density=0.05)
    #instances = ecole.instance.CombinatorialAuctionGenerator(n_items=50, n_bids=150)
    #instances = ecole.instance.IndependentSetGenerator(n_nodes=200)
    #instances = ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=15, n_facilities=15)

    #instances.seed(0)

    n_instances = 5

    solvers = [
        UBSolver(20, pool=pool),
        GnnSolver("agent_sb.pkl"),
        ClassicSolver("vanillafullstrong"),
        ClassicSolver("relpscost"),
    ]

    metrics = ["nnodes", "time"]

    evaluate_solvers(solvers, instances, n_instances, metrics)


def evaluate_solvers(solvers: [Solver], instances, n_instances, metrics):
    print(f"{'Instance':<15}" + "".join([f"{str(solver):<{15*len(metrics)}}" for solver in solvers]))
    print(f"{'':<15}" + len(solvers)*("".join([f"{metric:<15}" for metric in metrics])))
    output_control = utils.OutputControl()

    data = np.zeros((len(solvers), n_instances, len(metrics)))

    for i, instance in zip(range(n_instances), instances):
        output_control.mute()
        instance.as_pyscipopt().writeProblem("prob.lp")
        output_control.unmute()

        print(f"{i:^15}", end="")
        for j, solver in enumerate(solvers):
            solver.solve("prob.lp")
            line = [solver[metric] for metric in metrics]
            for k, d in enumerate(line):
                data[j, i, k] = d
            print("".join([f"{d:{'<15.3f' if type(d)==float else '<15'}}" for d in line]), end="")

        print()

    return data




if __name__ == "__main__":
    t = time.time()
    utils.mpi_start(main)
    t = time.time() - t
    print(f"Runned in {round(t,2)} s")