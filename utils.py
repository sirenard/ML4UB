import ecole
from boundml.components import StrongBranching, EcoleComponent
from boundml.components.gnn_component import GnnBranching
from boundml.instances import IndependentSetGenerator, SetCoverGenerator, CapacitatedFacilityLocationGenerator, \
    CombinatorialAuctionGenerator

from lb_component import LiftedBranching
from ucig_generator import UcigGenerator


def get_instances(name: str, difficulty: str, seed = None, start = None):
    match name:
        case "uc":
            days = {"easy": 1, "medium": 2, "hard": 3}
            if start is None:
                start = 0
            else:
                start = start // days[difficulty]
            print(f"Start at: {start}")
            instances = UcigGenerator("instances", days=days[difficulty], start_at_instance=start)
        case "ca":
            kwargs = {"easy": {"n_items": 100, "n_bids": 500}, "medium": {"n_items": 200, "n_bids": 1000},
                      "hard": {"n_items": 300, "n_bids": 1500}, }[difficulty]
            instances = CombinatorialAuctionGenerator(**kwargs)
        case "cfl":
            kwargs = {"easy": {"n_customers": 100}, "medium": {"n_customers": 200}, "hard": {"n_customers": 400}, }[
                difficulty]
            instances = CapacitatedFacilityLocationGenerator(**kwargs)
        case "cflfixed":
            kwargs = {"easy": {"n_customers": 100}, "medium": {"n_customers": 200}, "hard": {"n_customers": 400}, }[
                difficulty]
            instances = CapacitatedFacilityLocationGenerator(**kwargs, fixed_facilities=True)
        case "sc":
            kwargs = {"easy": {"n_rows": 500, "n_cols": 1000}, "medium": {"n_rows": 1000, "n_cols": 1000},
                      "hard": {"n_rows": 2000, "n_cols": 1000}, }[difficulty]
            instances = SetCoverGenerator(**kwargs)
        case "mis":
            kwargs = {"easy": {"n_nodes": 500}, "medium": {"n_nodes": 1000}, "hard": {"n_nodes": 1500}, }[
                difficulty]
            instances = IndependentSetGenerator(**kwargs)
        case _:
            raise NotImplementedError

    if name != "uc" and seed is not None:
        instances.seed(seed)

        if start is not None:
            for _ in range(start):
                next(instances)

    return instances

def get_component(name: str, usegpu=True, ncpu=12):
    if name == "sb":
        return StrongBranching(False, True, True, True)

    elif name == "lb_sb":
        return LiftedBranching(
            expert=StrongBranching(False, True, True, True),
            scip_params={},
            score_method="speedup",
            node_limit=100,
            local=True,
            n_proc=ncpu
        )
    elif name.startswith("lb_"):
        model_name = name.replace("lb_", "")
        strat = GnnBranching(
                model_name,
                feature_component = EcoleComponent(ecole.observation.NodeBipartite()),
                try_use_gpu=False
            )
        return LiftedBranching(
            expert=strat,
            scip_params={},
            score_method="speedup",
            node_limit=100,
            local=True,
            n_proc=ncpu
        )

    else:
        return GnnBranching(
            name,
            feature_component=EcoleComponent(ecole.observation.NodeBipartite()),
            try_use_gpu=usegpu
        )