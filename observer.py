import os
import pickle

import ecole
import pyscipopt
import numpy as np
from boundml.observers import Observer

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

