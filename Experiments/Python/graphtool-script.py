from graph_tool.all import *
import sys
import pandas as pd
import numpy as np

K = int(sys.argv[1])
id = sys.argv[2]

g1 = load_graph("temp/multilayer" + id + ".graphml")
state = minimize_blockmodel_dl(g1,multilevel_mcmc_args=dict(B_max=K,B_min=K),
                               state=LayeredBlockState, state_args=dict(ec=g1.ep.weight, layers=True))

#for i in range(1000): # this should be sufficiently large
#    state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

pd.DataFrame(state.get_blocks().get_array()).to_csv("temp/result" + id + ".csv")

