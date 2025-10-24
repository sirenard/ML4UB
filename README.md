# Lifted Branching

This code is the code used in the experiment of the paper "Lifted Branching: Learning to Improve Branching Strategies".

## Dependencies

This code mainly depends on [boundml](https://github.com/sirenard/boundml) which can be installed using `pip install boundml`. 
See boundml's documentation for more information on installation.

To work with unit comitment instances: `pip install git+https://github.com/grid-parity-exchange/Egret pyomo`

## Generate a training set

The first step is to generate a dataset using an expert. This expert can be either strong branching, or lifted branching
LB[B] with B being strong branching or a GNN model trained during a previous iteration. This is done with the file
`generate.py`

```
usage: generate.py [-h] [--sample_prefix SAMPLE_PREFIX] [--start_at START_AT] [--instances {ca,cfl,sc,mis,uc}] [--lsb] [--expert EXPERT] [--out OUT] --ncpu NCPU [--nodelimit NODELIMIT] [--maxinstances MAXINSTANCES]
                   [--local LOCAL]

options:
  -h, --help            show this help message and exit
  --sample_prefix SAMPLE_PREFIX
  --start_at START_AT
  --instances {ca,cfl,sc,mis,uc}
  --lsb                 Whether to use SB as expert
  --expert EXPERT       Which expert to use, if sb -> LB[SB] is the expert. If path -> LB[GNN(path)] is the expert
  --out OUT             Folder to save the results to
  --ncpu NCPU           Number of CPU cores to use for LB component
  --nodelimit NODELIMIT
                        Additional limits used while computing LB scores
  --maxinstances MAXINSTANCES
                        Maximum number of instances to use
  --local LOCAL
```

## Train a model

Once a dataset is generated, a GNN can be trained using the file `train.py` which take as argument the dataset folder 
and the path to the pkl file containing the torch model.

## Test the different strategies

Once all the models are trained, they can be tested against other strategies using the file `evaluate.py`

```
Solve instances using different solvers to compare them.

options:
  -h, --help            show this help message and exit
  --instances {ca,cfl,sc,mis,uc}
  --difficulty {easy,medium,hard}
  --out OUT             Where to save all the data to compute reports later
  --n_instances N_INSTANCES
                        Number of instances to solve
  --solvers [SOLVERS ...]
                        "Which custom solvers to use. sb -> Strong Branching lb_sb -> Lifted Branching of strong branching lb_path -> Lifted branching, using GNN branching using model in `path` path -> GNN branching using
                        model in `path`
  --classical_solvers [CLASSICAL_SOLVERS ...]
                        WHich default SCIP's strategies to used (relpscost, pscost, ...)
  --ncpu NCPU           Number of CPU cores to use. If no LB solver is used, ncpu instances are solved in parallel. If a LB solver is used, ncpu are used for computing LB scores
  --timelimit TIMELIMIT
                        Time limit in seconds for each instance
  --usegpu              Should GNN strategies use GPU
  --ntorchthreads NTORCHTHREADS
                        Number of torch threads to use
  --seed SEED
```

### Reporting results

The script `evaluate.py` saves all the metrics of each solving process in a pkl as a `SolverEvaluationResults` object from boundml.
If different solvers are evaluate in different run but on same instances and with same settings, these objects can be combined later
to compare all the solvers between each other.

Assume 4 solvers have been evaluated in 2 runs with results in `out1.pkl` and `out2.pkl`, a general report can be computed:

```python
res1: SolverEvaluationResults = pickle.load(open("out1", "rb"))
res2: SolverEvaluationResults = pickle.load(open("out2", "rb"))

res = res1 + res2 # results can be combined

# Compute a report with differents metrics
r = results.compute_report(
    SolverEvaluationResults.sg_metric("nnodes", 10),
    SolverEvaluationResults.sg_metric("time", 1),
    SolverEvaluationResults.nwins("time"),
    SolverEvaluationResults.nsolved(),
    SolverEvaluationResults.auc_score("time")
)

print(r)
```

## Evaluate accuracies

To compute the Acc@n accuracies of the model, the file `accuracy.py` can be used.

```
usage: accuracy.py [-h] [--instances {ca,cfl,sc,mis,uc}] [--n_instances N_INSTANCES] [--timelimit TIMELIMIT] [--seed SEED] [--ncpu NCPU] -p SOLVER_PAIR SOLVER_PAIR

Solve the instances using the model strategy. At each decision, compute expert scores and store how the model's decision is ranked according to expert scores. After solving all the instances, report the Acc@1 Acc@5 Acc@10 in
percent.

options:
  -h, --help            show this help message and exit
  --instances {ca,cfl,sc,mis,uc}
  --n_instances N_INSTANCES
  --timelimit TIMELIMIT
  --seed SEED
  --ncpu NCPU           Number of CPUs to use for LB components
  -p, --solver_pair SOLVER_PAIR SOLVER_PAIR
                        Pair of solvers: expert, model. The format is the same one as used for evaluation: sb -> Strong Branching lb_sb -> Lifted Branching of strong branching lb_path -> Lifted branching, using GNN branching
                        using model in `path` path -> GNN branching using model in `path`
```
