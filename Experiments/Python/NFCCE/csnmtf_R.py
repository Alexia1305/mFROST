# add current directory

import numpy as np
import pylab as pl
#import os, sys
#sys.path.append("/Users/vgligorijevic/Projects/NF-CCE/data/mostafavi/")

from snmtf import SNMTF
from csnmtf import CSNMTF
from preprocessing import nets_from_mat, mltplx_from_mat, net_normalize
from cluster import nmf_clust, spect_clust, clust_eval


def CSNMTF_algo(Nets,r,alpha=0.1,numTrials=3,seed=None):
    if seed is not None:
        np.random.seed(seed)
    Nets = net_normalize(Nets)

    objCSNMTF = CSNMTF(Nets, r, alpha = alpha,numRestarts=numTrials, init = 'rnda', displ = False,tol=1e-5)
    res_csnmtf = objCSNMTF.fit()
    H = (res_csnmtf.matrices[0])
   
    # return labels    
    return nmf_clust(H)

   
