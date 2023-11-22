import time
import ecole
import pyscipopt

import utils
from bpp import BinpackingGenerator
from environments import HistoryBranchingEnvironment, SimpleConfiguring
from observer import UnrealisticObserver

n_instances = 10
n_repetitions = 20
n_cpus = 4

policies = ["UB", "vanillafullstrong", "relpscost", "pscost", "random"]

if __name__ == "__main__":
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

    observer = UnrealisticObserver(
        n_repetitions,
        n_cpus,
        scip_params={
            "limits/nodes": 2000
        }
    )
    env = HistoryBranchingEnvironment(
        observation_function=(
            observer,
            ecole.observation.NodeBipartite(),
        ),
        information_function={
            "nb_nodes": ecole.reward.NNodes(),
            "time": ecole.reward.SolvingTime(),
        }
    )

    default_env = SimpleConfiguring(
        observation_function=None,
        information_function={
            "nb_nodes": ecole.reward.NNodes(),
            "time": ecole.reward.SolvingTime(),
        },
        scip_params={
            "branching/vanillafullstrong/priority": 99999999,
        },
    )

    sum_nodes = [0] * len(policies)
    output_control = utils.OutputControl()

    for i, instance in zip(range(n_instances), instances):
        print(f"instance {i+1}/{n_instances}")

        output_control.mute()
        instance.as_pyscipopt().writeProblem("prob.lp")
        output_control.unmute()

        observer.reset("prob.lp", 0)
        default_env.seed(0)
        env.seed(0)

        if "UB" in policies:
            ub_index = policies.index("UB")

            nb_nodes, t = 0, 0
            t = time.time()
            observation, action_set, _, done, info = env.reset(instance)
            nb_nodes += info["nb_nodes"]

            while not done:
                scores, is_expert = observation[0]
                action_index = scores[action_set].argmax()
                action = action_set[action_index]

                observer.step(action)
                observation, action_set, _, done, info = env.step(action)

                nb_nodes += info["nb_nodes"]

            sum_nodes[ub_index] += nb_nodes
            print(f"UB: {nb_nodes}")


        # Solve with ub
        for i, policy in enumerate(policies):
            if policy == "UB":
                continue

            model = pyscipopt.Model()
            utils.configure(model, branching_policy=policy)
            model.readProblem("prob.lp")
            nnodes, obj = utils.solve(model)
            sum_nodes[i] += nnodes
            print(f"{policy}: {nnodes}")


    print("******************")
    print("*****RESULTS******")
    print("******************")
    for policy, score in zip(policies, sum_nodes):
        print(f"{policy}: {score}")


