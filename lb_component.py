import os
import pickle
import threading

import numpy as np
import pyscipopt
from boundml.components import ScoringBranchingStrategy, StrongBranching
from pyscipopt import Model



class LiftedBranching(ScoringBranchingStrategy):
    tmp_observer = None

    def __init__(self, expert=StrongBranching(), scip_params={}, score_method=None,
                 node_limit=50, local=False, n_proc=24):
        super().__init__()
        self.scip_params = scip_params
        self.expert = expert
        self.score_method = score_method
        self.node_limit = node_limit

        self.is_child = False
        self.writing_pipe = None
        self.local = local
        self.n_proc = n_proc

    def compute_scores(self, model: Model):
        scores = super().compute_scores(model)
        if self.is_child:
            return self.expert.compute_scores(model)

        # List of branching candidates
        candidates, *_ = model.getLPBranchCands()

        pipes = [os.pipe() for _ in candidates]
        pids = []

        var: pyscipopt.Variable
        for i, var in enumerate(candidates):
            pid = os.fork()
            r, w = pipes[i]
            if pid > 0:  # parent
                pids.append(pid)
                os.close(w)
                if len(pids) == self.n_proc:
                    p = pids[0]
                    os.waitpid(p, 0)
                    pids.remove(p)
            else:  # child
                os.close(r)
                self.is_child = True
                self.writing_pipe = w

                model.setParam("limits/restarts", 1)
                model.setParam("presolving/maxrestarts", 0)

                if self.local:
                    leaves, _, _ = model.getOpenNodes()
                    for node in leaves:
                        model.updateNodeLowerbound(node, model.infinity())

                scores[i] = 1

                node_lim = model.getNNodes() + self.node_limit
                model.setParam("limits/nodes", node_lim)

                return scores

        # Wait for all children
        for pid in pids:
            os.waitpid(pid, 0)

        for i in range(len(candidates)):
            r, _ = pipes[i]

            r = os.fdopen(r, "rb")
            str = r.read()
            d = pickle.loads(str)
            score = d["nnodes"]

            if self.score_method == "custom":
                score = None

            scores[i] = score

            r.close()

        match self.score_method:
            case "n":
                scores = -scores / np.linalg.norm(scores)
            case "b":
                self.expert.compute_scores(model)
                expert_scores = self.expert.scores
                expert_action = expert_scores.argmax()
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


        return scores


    def reset(self, model: Model) -> None:
        self.expert.reset(model)

    def done(self, model: Model) -> None:
        self.expert.done(model)
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
        return f"lifted({str(self.expert)})"