import json
import math
import pickle
import tempfile
import os

import ecole
import numpy as np
import pyscipopt
import torch
from ecole.core.observation import NodeBipartite

import utils
from model import load_policy, get_device


class Observer:
    def __init__(self, seed=None, principal_observer=False):
        self.seed = seed
        self.instance_path = None
        self.principal_observer = principal_observer

    def before_reset(self, model):
        return

    def extract(self, model, done):
        return

    def reset(self, instance_path, seed=None):
        self.instance_path = instance_path
        self.seed = seed

    def done(self, model):
        return

    def set_principal_observer(self, val):
        self.principal_observer = val

    def is_principal_observer(self):
        return self.principal_observer

    def __str__(self):
        return "default"


class RandomObserver(Observer):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.rng = np.random.default_rng()

    def before_reset(self, model):
        return

    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = m.getLPBranchCands()
        n_vars = int(m.getNVars())

        prob_indexes = sorted([var.getCol().getLPPos() for var in candidates])

        scores = self.rng.random(len(prob_indexes))
        res = np.zeros(n_vars)
        res[:] = np.nan
        res[prob_indexes] = scores
        return res

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __str__(self):
        return "random"


class EntropyObserver(Observer):
    def __init__(self, n_solve=10):
        super().__init__()
        self.n_solve = n_solve
        self.variables_entropy = {}
        self.variables_distribution = {}
        self.values = {}
        self.solutions = []

    @staticmethod
    def entropy(xs):
        n = np.sum(xs)
        if n > 0:
            ps = xs / n
        else:
            ps = xs
        return -np.sum(np.log2(ps, where=(xs != 0)) * ps)

    def get_conditional_entropies(self, variables):
        res = []
        for name in variables:
            lb, ub = variables[name]
            dist = np.zeros(int(ub - lb) + 1)
            for sol in self.solutions:
                good = True
                for v in variables:
                    if not (sol[v] >= variables[v][0] and variables[v][1] <= ub):
                        good = False
                        break
                if good:
                    dist[int(sol[name] - lb)] += 1

            res.append(self.entropy(dist))

        return res

    def get_entropy(self, model):
        prob_file = tempfile.NamedTemporaryFile(suffix=".lp", dir="tmp")
        # m = pyscipopt.Model(sourceModel=model, globalcopy=False)
        output = utils.OutputControl()
        output.mute()
        model.writeProblem(prob_file.name)
        output.unmute()
        prob_name = prob_file.name

        vars_dist = {}
        n_sols = 0

        for var in model.getVars():
            # var: pyscipopt.scip.Variable
            if var.vtype() in ['BINARY', 'INTEGER']:
                name = var.name
                dists = np.zeros(int(var.getUbGlobal() - var.getLbGlobal() + 1))
                vars_dist[name] = dists

        m2 = pyscipopt.Model()
        m2.setParam("display/verblevel", 0)

        self.solutions = []
        for i in range(self.n_solve):
            m2.setParam("randomization/lpseed", i)
            m2.setParam("randomization/randomseedshift", i)
            m2.setParam("randomization/permutationseed", i)
            m2.readProblem(prob_name)
            m2.optimize()

            best_obj = m2.getObjVal()
            for sol in m2.getSols():
                # sol: pyscipopt.scip.Solution
                obj = m2.getSolObjVal(sol)
                if obj >= 1.5 * best_obj:
                    continue

                n_sols += 1
                s = str(sol).replace("'", "\"")
                self.solutions.append(json.loads(s))
                # for var in m2.getVars():
                #     # var: pyscipopt.scip.Variable
                #     if var.vtype() in ['BINARY', 'INTEGER']:
                #         name = var.name
                #         # s = str(sol)
                #         # s = s.replace("'", "\"")
                #         # val = json.loads(s)[name]
                #
                #         val = sol[var]
                #         val = int(round(val))
                #         lb = var.getLbGlobal()
                #         name = name.replace("t_", "")
                #         vars_dist[name][int(val - lb)] += 1

        prob_file.close()

        # print(n_sols)
        entropy = {}
        # for name in vars_dist:
        #     dist = vars_dist[name] #/ n_sols
        #     self.variables_distribution[name] = dist
        #     entropy[name] = self.entropy(dist)

        return entropy

    def before_reset(self, model):
        m: pyscipopt.Model = model.as_pyscipopt()
        self.variables_entropy = self.get_entropy(m)

    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        # entropy = self.get_entropy(m)
        # entropy = self.variables_entropy
        candidates, *_ = m.getLPBranchCands()
        n_vars = int(m.getNVars())

        res = np.zeros(n_vars)
        res[:] = np.nan

        vars = {}
        for var in m.getVars():
            # var: pyscipopt.scip.Variable
            if var.vtype() in ['BINARY', 'INTEGER']:
                name = var.name.replace("t_", "")
                # dist = self.variables_distribution[name]
                # lb = var.getLbGlobal()
                # start = int(var.getLbLocal() - lb)
                # end = int(var.getUbLocal() - lb)
                # e += self.entropy(dist[start:end + 1])
                lb = var.getLbLocal()
                ub = var.getUbLocal()
                vars[name] = [lb, ub]

        for var in candidates:
            var: pyscipopt.scip.Variable
            name = var.name
            name = name.replace("t_", "")
            prob_index = var.getCol().getLPPos()
            # dist = self.variables_distribution[name]
            # lb = var.getLbGlobal()
            # start = int(var.getLbLocal() - lb)
            # val = var.getLPSol()
            # end = int(var.getUbLocal() - lb)
            # e1 = self.entropy(dist[start:math.floor(val)+1])
            # e2 = self.entropy(dist[start+math.ceil(val):end+1])
            #
            # res[prob_index] = -(e1 + e2 + 2*(e - self.entropy(dist[start:end + 1])))
            # # res[prob_index] = (e1 + e2)

            lb, ub = vars[name]
            val = var.getLPSol()
            vars[name] = [lb, math.floor(val)]
            e1 = self.get_conditional_entropies(vars)
            vars[name] = [math.ceil(val), ub]
            e2 = self.get_conditional_entropies(vars)
            vars[name] = [lb, ub]
            res[prob_index] = (np.sum(e1) + np.sum(e2))

        return res

    def __str__(self):
        return "entropy"


class LiftedObserver(Observer):
    tmp_observer = None

    def __init__(self, observer=ecole.observation.StrongBranchingScores(), scip_params={}, score_method=None,
                 node_limit=50, local=False, n_proc=24):
        super().__init__()
        self.scip_params = scip_params
        self.observer = observer
        self.score_method = score_method
        self.node_limit = node_limit

        self.is_child = False
        self.writing_pipe = None
        self.local = local
        self.n_proc = n_proc

    def before_reset(self, model):
        self.observer.before_reset(model)

    def extract(self, model, done):
        if self.is_child:
            return self.observer.extract(model, done)

        m: pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = m.getLPBranchCands()
        n_vars = int(m.getNVars())

        prob_indexes = sorted([var.getCol().getLPPos() for var in candidates])

        pipes = [os.pipe() for _ in prob_indexes]
        pids = []

        # print(f"{len(candidates)} candidates")

        for i, index in enumerate(prob_indexes):
            pid = os.fork()
            r, w = pipes[i]
            if pid > 0:  #parent
                pids.append(pid)
                os.close(w)
                if len(pids) == self.n_proc:
                    p = pids[0]
                    os.waitpid(p, 0)
                    pids.remove(p)
            else:  #child
                os.close(r)
                self.is_child = True
                self.writing_pipe = w
                self.set_principal_observer(True)
                if self.local:
                    leaves, _, _ = m.getOpenNodes()
                    for node in leaves:
                        m.updateNodeLowerbound(node, m.infinity())
                res = np.zeros(n_vars)
                res[:] = np.nan
                res[prob_indexes] = 0
                res[index] = 1

                node_lim = m.getNNodes() + self.node_limit
                m.setParam("limits/nodes", node_lim)

                return res

        scores = np.zeros(len(prob_indexes))
        # print(f"{len(prob_indexes)=} candidates")

        for pid in pids:
            os.waitpid(pid, 0)
        for i in range(len(prob_indexes)):
            r, _ = pipes[i]

            # print("Child done")
            r = os.fdopen(r, "rb")
            str = r.read()
            d = pickle.loads(str)
            score = d["nnodes"]

            if self.score_method == "custom":
                score = None

            # print(f"Score {score}")

            scores[i] = score

            r.close()

        # print(scores)
        match self.score_method:
            case "n":
                scores = -scores / np.linalg.norm(scores)
            case "b":
                expert_scores = self.observer.extract(model, done)
                expert_action = expert_scores[prob_indexes].argmax()
                scores = (scores >= scores[expert_action]).astype(int)
            case "speedup":
                # expert_scores = self.observer.extract(model, done)
                # expert_action = expert_scores[prob_indexes].argmax()
                # expert_value = scores[expert_action]
                scores = np.log2((scores.min() + 1) / (scores + 1))
            case "custom":
                pass
            case _:
                scores = -scores

        res = np.zeros(n_vars)
        res[:] = np.nan
        res[prob_indexes] = scores
        return res

    def done(self, model):
        if not self.is_child:
            return

        if model.getGap() == 0:
            nnodes = model.getNNodes()
        else:
            nnodes = model.getTreesizeEstimation()


        w = os.fdopen(self.writing_pipe, 'wb')
        w.write(pickle.dumps({"nnodes": nnodes, "obj": model.getObjVal()}))
        w.close()
        os._exit(os.EX_OK)

    def __str__(self):
        return f"lifted({str(self.observer)})"


class StrongBranching(Observer):
    def __init__(self):
        super().__init__()
        self.observer = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        self.observer.before_reset(model)

    def extract(self, model, done):
        return self.observer.extract(model, done)

    def __getstate__(self):
        return []

    def __setstate__(self, _):
        self.__init__()

    def __str__(self):
        return "SB"


class GnnObserver(Observer):
    def __init__(self, policy_path: str, feature_observer=ecole.observation.NodeBipartite(), try_use_gpu=False,
                 **kwargs):
        super().__init__()
        self.policy_path = policy_path
        self.policy = load_policy(policy_path, try_use_gpu, **kwargs)
        self.feature_observer = feature_observer
        self.policy_path = policy_path
        self.try_use_gpu = try_use_gpu
        self.has_tree_features = "n_tree_features" in kwargs

    def before_reset(self, model):
        self.feature_observer.before_reset(model)

    def extract(self, model, done):
        # print(f"{os.getpid()}: Extract")
        device = get_device(self.try_use_gpu)
        observation = self.feature_observer.extract(model, done)
        observation= (
            torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
            torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
            torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device),
            torch.from_numpy(observation.variable_features.astype(np.float32)).to(device),
            observation.variable_features.shape[0],
            torch.from_numpy(observation.tree_features.astype(np.float32)).to(
                device) if self.has_tree_features else np.array([]),
        )

        logits = self.policy(*observation)
        return logits.cpu().detach().numpy()

    def __getstate__(self):
        return self.policy_path

    def __setstate__(self, state):
        self.__init__(state)

    def __str__(self):
        return f"GNN({self.policy_path})"


class LpObserver(Observer):
    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        return m.getLPObjVal()


class NodeBipartiteObsExtended:
    def __init__(self, node, tree_features=np.array([])):
        self.RowFeatures = node.RowFeatures
        self.VariableFeatures = node.VariableFeatures

        self.edge_features = node.edge_features
        self.row_features = node.row_features
        self.variable_features = node.variable_features
        self.tree_features = tree_features


class NodeBipartiteExtended(NodeBipartite):
    def __init__(self):
        super().__init__()
        self.gap = 0
        self.leaf_frequency = 0
        self.open_nodes = 0
        self.ssg = 0
        self.tree_weight = 0

    def before_reset(self, model):
        super().before_reset(model)

        self.gap = 0
        self.leaf_frequency = 0
        self.open_nodes = 0
        self.ssg = 0
        self.tree_weight = 0

    def extract(self, model, done):
        obs = super().extract(model, done)

        m: pyscipopt.Model = model.as_pyscipopt()

        nnodes = m.getNNodes()
        self.gap = m.getGap()

        self.leaf_frequency = 1 / nnodes * (m.getNLeaves() - 1 / 2)

        completion = m.getNNodes() / m.getTreesizeEstimation()
        tree_features = np.array([completion])
        return NodeBipartiteObsExtended(obs, tree_features)


class DistObserver(Observer):
    def __init__(self, expert_probability=1, seed=0):
        super().__init__(seed)
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.sol = None
        self.rng = np.random.default_rng(seed)

    def extract(self, model, done):
        is_expert = self.rng.random() <= self.expert_probability
        if is_expert:
            scores = self.extract_expert(model)
        else:
            scores = self.pseudocosts_function.extract(model, done)
            scores = np.array(scores)

        return scores, is_expert

    def extract_expert(self, model):
        pyscipopt_model: pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = pyscipopt_model.getLPBranchCands()
        n_actions = len(candidates)
        n_vars = int(pyscipopt_model.getNVars())

        prob_indexes = sorted([var.getCol().getLPPos() for var in candidates])
        scores = []

        prob_file = tempfile.NamedTemporaryFile(suffix=".lp", dir="tmp")
        output = utils.OutputControl()
        output.mute()
        pyscipopt_model.writeProblem(prob_file.name, trans=True)
        output.unmute()

        for index, var in zip(prob_indexes, candidates):
            var: pyscipopt.scip.Variable
            lp_val = var.getLPSol()
            distances = []
            models = []
            for i in range(2):
                m = pyscipopt.Model()
                m.setParam("display/verblevel", 0)
                m.readProblem(prob_file.name)
                for v in m.getVars():
                    m.chgVarType(v, 'CONTINUOUS')
                    if v.name == var.name:
                        if i == 0:
                            m.tightenVarUbGlobal(v, math.floor(lp_val))
                        else:
                            m.tightenVarLbGlobal(v, math.ceil(lp_val))

                m.optimize()
                distances.append(self.get_dist(m))
                models.append(m)

            # dist=abs(distances[0]-distances[1])
            # dist = self.get_models_dist(*models) - min(distances)
            # dist = self.get_models_dist(*models)
            if np.inf in distances:
                distances.remove(np.inf)
                dist = distances[0]
            else:
                dist = abs(distances[0] - distances[1])
            scores.append(dist)

        prob_file.close()

        res = np.zeros(n_vars)
        res[:] = np.nan
        res[prob_indexes] = scores
        return res

    def before_reset(self, model):
        self.pseudocosts_function.before_reset(model)

        m = pyscipopt.Model()
        m.setParam("display/verblevel", 0)
        m.readProblem(self.instance_path)
        m.optimize()
        best_sol = m.getBestSol()

        self.sol = {}
        for var in m.getVars():
            var: pyscipopt.scip.Variable
            t_var = m.getTransformedVar(var)

            val = best_sol[var]

            self.sol[t_var.name] = val

    def get_dist(self, model: pyscipopt.Model):
        if model.getStatus() != "optimal":
            return np.inf

        sol = model.getBestSol()
        dist = 0
        for var in model.getVars():
            t_var = model.getTransformedVar(var)
            if var.name in self.sol:
                dist += (sol[t_var] - self.sol[var.name]) ** 2

        return dist ** 0.5

    def get_models_dist(self, m1: pyscipopt.Model, m2: pyscipopt.Model):
        if m1.getStatus() != "optimal" or m2.getStatus() != "optimal":
            return np.inf

        vars = {}

        for m in (m1, m2):
            sol = m.getBestSol()
            for var in m.getVars():
                name = var.name
                if name not in vars:
                    vars[name] = []

                vars[name].append(sol[var])

        dist = 0
        for var in vars:
            dist += (vars[var][0] - vars[var][1]) ** 2

        return dist ** 2

    def __str__(self):
        return "dist"


class SVMObserver(Observer):
    def __init__(self, rank_svm_path, feature_observer=ecole.observation.Khalil2016()):
        super().__init__()
        with open(rank_svm_path, "rb") as f:
            self.model = pickle.load(f)

        self.feature_observer = feature_observer

    def before_reset(self, model):
        self.feature_observer.before_reset(model)

    def extract(self, model, done):
        m = model.as_pyscipopt()
        n_vars = int(m.getNVars())
        candidates, *_ = m.getLPBranchCands()

        prob_indexes = [var.getCol().getLPPos() for var in candidates]
        features = self.feature_observer.extract(model, done)
        X = np.nan_to_num(features.features[prob_indexes])
        indexes = self.model.predict(X)
        scores = [0 for _ in candidates]
        for i, j in enumerate(indexes):
            scores[j] = len(indexes) - i

        res = np.zeros(n_vars)
        res[:] = np.nan
        res[prob_indexes] = scores

        return res


class LinearObserver(Observer):
    def __init__(self, model_path, feature_observer=ecole.observation.Khalil2016()):
        super().__init__()
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.feature_observer = feature_observer

    def before_reset(self, model):
        self.feature_observer.before_reset(model)

    def extract(self, model, done):
        m = model.as_pyscipopt()
        n_vars = int(m.getNVars())
        candidates, *_ = m.getLPBranchCands()

        prob_indexes = [var.getCol().getLPPos() for var in candidates]
        features = self.feature_observer.extract(model, done)
        X = np.nan_to_num(features.features[prob_indexes])
        scores = self.model.predict(X)

        res = np.zeros(n_vars)
        res[:] = np.nan
        res[prob_indexes] = scores

        return res


class AccuracyObserver(Observer):
    """
    Given 2 observers, this observer extracts the placement of the second observer choice in the scoring of the first one
    It stores all its observations
    """
    def __init__(self, oracle: Observer, model_observer: Observer):
        super().__init__()
        self.oracle: Observer = oracle
        self.model_observer: Observer = model_observer
        self.observations = []

    def before_reset(self, model):
        self.oracle.before_reset(model)
        self.model_observer.before_reset(model)

    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = m.getLPBranchCands()
        prob_indexes = [var.getCol().getLPPos() for var in candidates]

        oracle_scores = self.oracle.extract(model, done)

        if self.oracle.is_principal_observer():
            self.set_principal_observer(True)
            return oracle_scores

        oracle_scores = oracle_scores[prob_indexes]

        model_scores = self.model_observer.extract(model, done)
        model_scores = model_scores[prob_indexes]

        best_index = np.argmax(model)
        oracle_sorted_indexes = np.argsort(-oracle_scores)

        position = np.where(oracle_sorted_indexes == best_index)[0][0] + 1
        self.observations.append(position)
        return position

    def get_observations(self):
        return np.array(self.observations)

    def done(self, model):
        self.oracle.done(model)
        self.model_observer.done(model)

    def reset(self, instance_path, seed=None):
        super().reset(instance_path, seed)
        self.oracle.reset(instance_path, seed)
        self.model_observer.reset(instance_path, seed)

    def __str__(self):
        return f"Acc {str(self.model_observer)}"

class ConditionalObservers(Observer):
    def __init__(self, observers, conditions):
        super().__init__()
        self.observers = observers
        self.conditions = conditions
        self.last_observer_index_used = -1

    def before_reset(self, model):
        for observer in self.observers:
            if observer is not None:
                observer.before_reset(model)

    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        for i, (observer, condition) in enumerate(zip(self.observers, self.conditions)):
            if condition(m):
                self.last_observer_index_used = i
                if observer is None:
                    return
                else:
                    return observer.extract(model, done)


    def reset(self, instance_path, seed=None):
        super().reset(instance_path, seed)
        for observer in self.observers:
            if isinstance(observer, Observer):
                observer.reset(instance_path, seed)

    def get_last_observer_index_used(self):
        return self.last_observer_index_used

    def done(self, model: pyscipopt.Model):
        for observer in self.observers:
            if isinstance(observer, Observer):
                observer.done(model)

    def __str__(self):
        s = ", ".join([str(observer) for observer in self.observers])
        return f"Cond({s})"
