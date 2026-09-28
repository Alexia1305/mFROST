import sys
import random
import numpy as np
import pandas as pd
import time 
from graph_tool.all import (
    seed_rng, load_graph, minimize_blockmodel_dl, LayeredBlockState
)

K = int(sys.argv[1])
id = sys.argv[2]
seed = int(sys.argv[3])
numTrials = int(sys.argv[4])

if numTrials < 1:
    raise ValueError("numTrials doit être supérieur ou égal à 1.")

seed_rng(seed)
np.random.seed(seed)
random.seed(seed)

g1 = load_graph(f"temp/multilayer{id}.graphml")

best_score = np.inf
best_blocks = None
for trial in range(numTrials):
    state = minimize_blockmodel_dl(
        g1,
        multilevel_mcmc_args=dict(B_min=K, B_max=K),
        state=LayeredBlockState,
        state_args=dict(ec=g1.ep.weight, layers=True)
    )

    score = state.entropy()

    if score < best_score:
        best_score = score
        best_blocks = state.get_blocks().get_array().copy()

if best_blocks is None:
    raise RuntimeError("No sucessful trial")

pd.DataFrame(best_blocks).to_csv(f"temp/result{id}.csv")