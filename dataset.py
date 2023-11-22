import gzip
import pickle
from pathlib import Path
import ecole

from environments import HistoryBranchingEnvironment
from observer import UnrealisticObserver, ExploreThenStrongBranch


def data_generation(instances, data_max_samples, folder_name = "samples", scip_params={}, **kwargs):
    episode_counter, sample_counter = 0, 0
    Path("{}/".format(folder_name)).mkdir(exist_ok=True)

    observer = UnrealisticObserver(scip_params=scip_params, **kwargs)
    env = HistoryBranchingEnvironment(
        observation_function=(
            #ExploreThenStrongBranch(**kwargs),
            observer,
            ecole.observation.NodeBipartite(),
        ),
        information_function={
            "nb_nodes": ecole.reward.NNodes(),
            "time": ecole.reward.SolvingTime(),
        },
        scip_params=scip_params
    )

    # We will solve problems (run episodes) until we have saved enough samples
    while sample_counter < data_max_samples:
        episode_counter += 1

        env.seed(0)
        instance = next(instances)

        instance.as_pyscipopt().writeProblem("prob.lp")
        observer.reset("prob.lp", 0)

        observation, action_set, _, done, _ = env.reset(instance)

        while not done:
            (scores, scores_are_expert), node_observation = observation
            action_index = scores[action_set].argmax()
            action = action_set[action_index]
            # Only save samples if they are coming from the expert (strong branching)
            if scores_are_expert and (sample_counter < data_max_samples):
                sample_counter += 1
                data = [node_observation, action, action_set, scores]
                filename = f"{folder_name}/sample_{sample_counter}.pkl"

                with gzip.open(filename, "wb") as f:
                    pickle.dump(data, f)

            observer.step(action)
            observation, action_set, _, done, _ = env.step(action)

        print(f"Episode {episode_counter}, {sample_counter} samples collected so far")

def get_sample_files(folder: str):
    return [str(path) for path in Path("{}/".format(folder)).glob("sample_*.pkl")]